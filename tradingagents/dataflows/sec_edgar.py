"""Company statements as they were filed, from SEC EDGAR.

Every other fundamentals vendor serves a period's current value and cuts the
statement at the fiscal period end. That is two claims a run should not make: a
period that has ended is not public until the company files, weeks later, and a
figure that was later restated is not what investors saw at the time.

EDGAR reports every fact with the date it was filed, so a run dated ``curr_date``
serves exactly what was on file by then, restatements included at the vintage
that was current: Apple's 2008 total assets read 39.6B until the 2010 amendment
restated them to 36.2B.

The same rule gives a past date its valuation. Vendor profiles carry today's
market cap and multiples, so they are withheld from a historical run (#1300) and
the analyst is left with statements and no price attached to them.
``get_fundamentals`` rebuilds the multiples from what was public on the date: the
share count and trailing figures on file by then, and that day's close.

Access needs no key or account, only a User-Agent identifying the caller, which
SEC requires and refuses requests without. US filers only: anything absent from
EDGAR's ticker map falls through to the next configured vendor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from importlib import metadata
from pathlib import Path

import requests

from .config import get_config
from .errors import NoMarketDataError, VendorRateLimitError
from .stockstats_utils import yf_retry
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# A filing history only changes when something new is filed, so one fetch per
# company per day serves every date a run asks about.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A year to date, longest first: nine months, six, then the first quarter alone.
_YEAR_TO_DATE = ((240, 295), (150, 205), (60, 115))

# Lines a valuation needs beyond the statement rows above, same order rule.
_VALUATION_TAGS: dict[str, tuple[str, ...]] = {
    "Depreciation and Amortization": ("DepreciationDepletionAndAmortization",
                                      "DepreciationAmortizationAndAccretionNet",
                                      "DepreciationAndAmortization"),
    "Marketable Securities": ("MarketableSecuritiesCurrent",
                              "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                              "ShortTermInvestments"),
}
_VALUATION_TAGS.update({label: tags for rows in _STATEMENTS.values() for label, tags in rows})

# Borrowings are separate lines that add up, unlike alternative tags for one line.
_DEBT_LINES = ("LongTermDebtNoncurrent", "LongTermDebtCurrent", "CommercialPaper",
               "ShortTermBorrowings")

# The newest filed period can trail the date by a quarter plus a filing deadline.
# Anything older means the company stopped filing, and a multiple would mislead.
_STALE_AFTER_DAYS = 200


def _user_agent() -> str:
    """Who SEC sees. No account or key exists; callers identify themselves.

    www.sec.gov, which serves the ticker map, refuses a User-Agent carrying no
    contact address: a client name alone or with a project URL gets 403, one
    with an address gets 200. So the default carries a placeholder address and
    the package version. Set SEC_EDGAR_USER_AGENT to your own name and address
    so SEC can reach you about your traffic rather than the project.
    """
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{_version()} (contact@example.com)"


def _version() -> str:
    """The installed package version, so a release identifies itself correctly."""
    try:
        return metadata.version("tradingagents")
    except metadata.PackageNotFoundError:
        return "dev"


def _fetch_json(url: str) -> dict:
    """Read a public EDGAR document, respecting SEC's identification rule."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Every failure here is "this vendor cannot serve it now", so the router
        # moves on instead of seeing a transport exception it has no rule for.
        raise VendorRateLimitError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorRateLimitError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    data = _fetch_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temp, path)
    return data


def cik_for(ticker: str) -> str | None:
    """The filer's CIK, or None when the ticker is not a US filer."""
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    wanted = ticker.strip().upper()
    for entry in table.values():
        if entry.get("ticker", "").upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    return None


def _as_of(facts: dict, tags: tuple[str, ...], curr_date: str, span: tuple[int, int]) -> tuple[dict, str]:
    """({period end: value}, unit) for the first tag the filer reports, as known then.

    A period reported more than once takes its latest filing on or before the
    date, so an amendment counts from the day it was filed and not before. The
    unit comes from the filing: most lines are USD, earnings per share are
    USD/shares, and scaling those alike would print a real figure as zero.
    """
    low, high = span
    values: dict[str, float] = {}
    chosen_unit = "USD"
    # Tags are tried in order and a period keeps the first one that reports it:
    # filers renamed lines over the years, so one tag covers only part of the
    # history. Values are never added across tags, which would double count.
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[str, dict] = {}
            for fact in unit_values:
                if fact["filed"] > curr_date or fact["end"] in values:
                    continue
                # A duration fact (revenue, cash flow) must cover the span asked
                # for. An instant fact (a balance) has no span and serves both.
                if "start" in fact:
                    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
                    if not low <= days <= high:
                        continue
                seen = latest.get(fact["end"])
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[fact["end"]] = fact
            if latest:
                chosen_unit = unit
                values.update({end: fact["val"] for end, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def _statement(kind: str, ticker: str, freq: str, curr_date: str, title: str) -> str:
    curr_date = curr_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")

    span = _SPANS["quarterly" if freq.lower() == "quarterly" else "annual"]
    lines = {label: _as_of(us_gaap, tags, curr_date, span) for label, tags in _STATEMENTS[kind]}
    periods = sorted({end for values, _ in lines.values() for end in values})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {curr_date}")

    header = (
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {curr_date}, at the values filed then\n\n"
    )
    rows = [",".join([""] + periods)]
    for label, (values, unit) in lines.items():
        # Every row spans the same columns, or a reader lines the table up wrong.
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if unit == "USD" else f"{label} ({unit})"
        # Plain numbers: a thousands separator would split the CSV field.
        cells = [
            (f"{values[p] / 1e6:.0f}" if unit == "USD" else f"{values[p]:.2f}")
            if p in values else "" for p in periods
        ]
        rows.append(",".join([name] + cells))
    return header + "\n".join(rows) + "\n"


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Balance sheet as filed on or before ``curr_date``."""
    return _statement("balance_sheet", ticker, freq, curr_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Income statement as filed on or before ``curr_date``.

    A fourth quarter is never derived: filers report it only inside the annual
    figure, and subtracting three separately filed quarters would invent a number
    with no filing date behind it.
    """
    return _statement("income_statement", ticker, freq, curr_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Cash flow statement as filed on or before ``curr_date``."""
    return _statement("cashflow", ticker, freq, curr_date, "Cash Flow Statement")


def _days_between(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _trailing_year(facts: dict, tags: tuple[str, ...], curr_date: str) -> dict | None:
    """Twelve months to the newest filed period end, from figures on file by the date.

    Right after an annual report the fiscal year is the trailing year. Later it is
    that fiscal year, plus the year to date, minus the same stretch a year before.
    Each of the three was filed on or before the date and is named in the result,
    so the sum can be checked against the filings. A missing part returns None: a
    trailing year is never estimated.
    """
    annual, _ = _as_of(facts, tags, curr_date, _SPANS["annual"])
    if not annual:
        return None
    year_end = max(annual)
    for span in _YEAR_TO_DATE:
        to_date, _ = _as_of(facts, tags, curr_date, span)
        newest = max((end for end in to_date if end > year_end), default=None)
        if newest is None:
            continue
        # The stretch has to start where the fiscal year stopped. A second quarter
        # reported alone would otherwise be added to a full year as if it were six months.
        if not span[0] <= _days_between(year_end, newest) <= span[1]:
            continue
        year_before = [end for end in to_date if abs(_days_between(end, newest) - 365) <= 10]
        if not year_before:
            return None
        prior = year_before[-1]
        months = round(_days_between(year_end, newest) / 30.4)
        return {
            "value": annual[year_end] + to_date[newest] - to_date[prior],
            "end": newest,
            "parts": (f"fiscal year to {year_end} ({annual[year_end] / 1e6:.0f}) + {months} months "
                      f"to {newest} ({to_date[newest] / 1e6:.0f}) - {months} months to {prior} "
                      f"({to_date[prior] / 1e6:.0f})"),
        }
    return {"value": annual[year_end], "end": year_end, "parts": f"fiscal year to {year_end}"}


def _newest(facts: dict, tags: tuple[str, ...], curr_date: str) -> tuple[str, float] | None:
    """(date, value) of the newest balance filed on or before the date."""
    values, _ = _as_of(facts, tags, curr_date, (0, 0))
    return max(values.items()) if values else None


def _shares_outstanding(all_facts: dict, curr_date: str) -> tuple[str, float] | None:
    """The newest share count on file, from the filing cover page where there is one.

    A company with several share classes reports each class separately and has no
    single cover-page total, so its count there goes stale; the balance sheet
    count and then the diluted average stand in.
    """
    us_gaap = all_facts.get("us-gaap") or {}
    sources = (
        (all_facts.get("dei") or {}, ("EntityCommonStockSharesOutstanding",), (0, 0)),
        (us_gaap, ("CommonStockSharesOutstanding",), (0, 0)),
        (us_gaap, ("WeightedAverageNumberOfDilutedSharesOutstanding",), _SPANS["quarterly"]),
    )
    for facts, tags, span in sources:
        values, _ = _as_of(facts, tags, curr_date, span)
        if values and _days_between(max(values), curr_date) <= _STALE_AFTER_DAYS:
            return max(values.items())
    return None


def _close_and_splits(ticker: str, curr_date: str) -> tuple[float, str, dict[str, float]]:
    """(close, session date, {split date: ratio}) for the last session on or before the date.

    Yahoo restates every past close for later splits, so the close it serves for a
    past date is not the price that traded. Later splits are undone here, which
    puts the price on the share basis of the date, the basis the filed share count
    is on.
    """
    import yfinance as yf

    symbol = normalize_symbol(ticker)
    handle = yf.Ticker(symbol)
    day = date.fromisoformat(curr_date)
    history = yf_retry(lambda: handle.history(start=(day - timedelta(days=10)).isoformat(),
                                              end=(day + timedelta(days=1)).isoformat(),
                                              auto_adjust=False))
    if history is None or history.empty or history["Close"].dropna().empty:
        raise NoMarketDataError(ticker, symbol, f"no closing price on or before {curr_date}")
    closes = history["Close"].dropna()
    session = closes.index[-1].date().isoformat()
    splits = {when.date().isoformat(): float(ratio)
              for when, ratio in yf_retry(lambda: handle.splits).items() if ratio}
    close = float(closes.iloc[-1])
    for when, ratio in splits.items():
        if when > session:
            close *= ratio
    return close, session, splits


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Valuation as it stood on ``curr_date``: that day's close against figures filed by then.

    Name, sector, beta, dividend yield and forward estimates have no filed
    vintage, so they are not served; the report says so rather than leaving a gap
    the agent might fill from memory.
    """
    curr_date = curr_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    all_facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json").get("facts") or {}
    us_gaap = all_facts.get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")
    shares = _shares_outstanding(all_facts, curr_date)
    if shares is None:
        raise NoMarketDataError(ticker, ticker, f"no share count filed near {curr_date}")

    close, session, splits = _close_and_splits(ticker, curr_date)
    counted_on, share_count = shares
    # A split between the count and the date changes the count, not the company.
    for when, ratio in splits.items():
        if counted_on < when <= session:
            share_count *= ratio
    market_cap = close * share_count

    def trailing(tags: tuple[str, ...]) -> dict | None:
        item = _trailing_year(us_gaap, tags, curr_date)
        return item if item and _days_between(item["end"], curr_date) <= _STALE_AFTER_DAYS else None

    flows = {label: trailing(_VALUATION_TAGS[label])
             for label in ("Revenue", "Net Income", "Operating Income",
                           "Depreciation and Amortization", "Operating Cash Flow",
                           "Capital Expenditure")}
    # Some filers tag depreciation alone and amortization elsewhere. EBITDA built on
    # it runs a little low, so the row says which figure it used.
    depreciation_only = ""
    if flows["Depreciation and Amortization"] is None:
        flows["Depreciation and Amortization"] = trailing(("Depreciation",))
        if flows["Depreciation and Amortization"]:
            depreciation_only = " (depreciation only, this filer tags amortization separately)"

    def flow(label: str) -> float | None:
        return flows[label]["value"] if flows[label] else None

    def same_period(*labels: str) -> bool:
        return all(flows[x] for x in labels) and len({flows[x]["end"] for x in labels}) == 1

    # One balance sheet date for every balance, so cash from one quarter is never
    # netted against debt from another.
    balance = _newest(us_gaap, _VALUATION_TAGS["Total Assets"], curr_date)
    balance_date = balance[0] if balance else None
    if balance_date and _days_between(balance_date, curr_date) > _STALE_AFTER_DAYS:
        balance_date = None

    def at_balance_date(tags: tuple[str, ...]) -> float | None:
        found = _newest(us_gaap, tags, curr_date)
        return found[1] if found and found[0] == balance_date else None

    equity = at_balance_date(_VALUATION_TAGS["Stockholders Equity"])
    cash = at_balance_date(_VALUATION_TAGS["Cash and Equivalents"])
    securities = at_balance_date(_VALUATION_TAGS["Marketable Securities"])
    debt_lines = {tag: at_balance_date((tag,)) for tag in _DEBT_LINES}
    debt_lines = {tag: value for tag, value in debt_lines.items() if value is not None}
    if not debt_lines:
        total = at_balance_date(("LongTermDebt",))
        debt_lines = {"LongTermDebt": total} if total is not None else {}
    debt = sum(debt_lines.values())
    enterprise = market_cap + debt - cash - (securities or 0) if cash is not None else None
    ebitda = (flow("Operating Income") + flow("Depreciation and Amortization")
              if same_period("Operating Income", "Depreciation and Amortization") else None)
    free_cash = (flow("Operating Cash Flow") - flow("Capital Expenditure")
                 if same_period("Operating Cash Flow", "Capital Expenditure") else None)

    def multiple(top: float | None, bottom: float | None) -> str:
        if top is None or bottom is None:
            return "unavailable (an input was not on file)"
        if bottom <= 0:
            return "not meaningful (the denominator is zero or negative)"
        return f"{top / bottom:.1f}x"

    def millions(value: float | None) -> str:
        return "unavailable" if value is None else f"{value / 1e6:.0f}"

    adjusted = "" if share_count == shares[1] else ", restated for the split since"
    debt_note = " + ".join(debt_lines) if debt_lines else "no borrowing line tagged, counted as none"
    lines = [
        f"# Company Fundamentals for {ticker.upper()}, USD in millions unless the row says otherwise",
        f"# Point-in-time as of: {curr_date}",
        f"# SEC EDGAR facts filed on or before {curr_date}, priced at the {session} close",
        "# Name, sector, beta, dividend yield and forward estimates have no filed vintage",
        "# and are not served.",
        "",
        f"Price (USD per share): {close:.2f}",
        f"Shares Outstanding (millions): {share_count / 1e6:.1f} (counted {counted_on}{adjusted})",
        f"Market Cap: {millions(market_cap)}",
        f"Enterprise Value: {millions(enterprise)} (debt: {debt_note})",
        f"PE Ratio (TTM): {multiple(market_cap, flow('Net Income'))}",
        f"Price to Sales (TTM): {multiple(market_cap, flow('Revenue'))}",
        f"Price to Book: {multiple(market_cap, equity)}",
        f"EV to EBITDA (TTM): {multiple(enterprise, ebitda)}",
        f"EV to Sales (TTM): {multiple(enterprise, flow('Revenue'))}",
        "Free Cash Flow Yield (TTM): " + (
            f"{free_cash / market_cap:.2%}" if free_cash is not None
            else "unavailable (an input was not on file)"),
        "",
        "# Trailing twelve months, each built only from filed figures",
    ]
    for label, item in flows.items():
        lines.append(f"{label} (TTM): " + (f"{millions(item['value'])} = {item['parts']}" if item
                                           else "unavailable (a part was not on file)"))
    lines += [
        f"EBITDA (TTM): {millions(ebitda)}{depreciation_only if ebitda is not None else ''}",
        f"Free Cash Flow (TTM): {millions(free_cash)}",
        "",
        f"# Balances at {balance_date or 'no recent balance sheet'}",
        f"Stockholders Equity: {millions(equity)}",
        f"Cash and Equivalents: {millions(cash)}",
        f"Marketable Securities: {millions(securities)}",
        f"Debt: {millions(debt) if debt_lines else 'none tagged'}",
    ]
    return "\n".join(lines) + "\n"
