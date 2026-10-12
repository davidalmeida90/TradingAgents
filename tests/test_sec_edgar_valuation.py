"""SEC EDGAR valuation: a past run gets a market cap and multiples as of its own date.

Vendor profiles carry today's market cap and multiples, so a historical run gets
none (#1300). The valuation is rebuilt from what was public on the date: the
share count and the figures filed by then, against that day's close. The traps
are splits: Yahoo restates past closes for later splits, while a filed share
count stays on the basis of its own date.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.vendors import sec_edgar

TICKER_MAP = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA Corp"}}


def _fact(end, val, filed, form="10-K", start=None):
    fact = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        fact["start"] = start
    return fact


def _facts(cover=None, balance_count=None, net_income=-1, extra=None):
    us_gaap = {
        "NetIncomeLoss": {"units": {"USD": [
            _fact("2024-01-28", 29_760_000_000 if net_income == -1 else net_income, "2024-02-21",
                  start="2023-01-30"),
            # A 10-Q reporting twelve months passes the span check but is no fiscal year.
            _fact("2024-04-28", 42_600_000_000, "2024-05-29", form="10-Q", start="2023-04-30"),
        ]}},
        "Revenues": {"units": {"USD": [_fact("2024-01-28", 60_922_000_000, "2024-02-21", start="2023-01-30")]}},
        "StockholdersEquity": {"units": {"USD": [
            _fact("2024-01-28", 42_978_000_000, "2024-02-21"),
            _fact("2024-04-28", 49_142_000_000, "2024-05-29", form="10-Q"),
        ]}},
    }
    if balance_count:
        us_gaap["CommonStockSharesOutstanding"] = {"units": {"shares": [balance_count]}}
    us_gaap.update(extra or {})
    facts = {"us-gaap": us_gaap}
    if cover:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}}}
    return {"cik": 1045810, "facts": facts}


COVERS = [
    _fact("2024-02-16", 2_464_000_000, "2024-02-21"),            # before the 10:1 split of 2024-06-10
    _fact("2024-05-24", 2_460_000_000, "2024-05-29", form="10-Q"),
]


@pytest.fixture
def filings(tmp_path, monkeypatch):
    """Serve the given company facts offline and price every date at a fixed close."""
    def serve(facts, close=946.30, session="2024-05-15", splits=None):
        monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
        monkeypatch.setattr(sec_edgar, "_fetch_json",
                            lambda url: TICKER_MAP if "company_tickers" in url else facts)
        monkeypatch.setattr(sec_edgar, "_close", lambda ticker, day: (close, session, splits or {}))
    return serve


def _row(report: str, name: str) -> str:
    return next(line for line in report.splitlines() if line.startswith(name))


@pytest.mark.unit
def test_a_past_date_gets_its_market_cap_and_multiples(filings):
    filings(_facts(cover=COVERS))
    report = sec_edgar.get_fundamentals("NVDA", "2024-05-15")
    # 946.30 x 2,464M shares counted on the 10-K cover; the 10-Q filed 2024-05-29 is not known yet.
    assert _row(report, "Market Cap") == "Market Cap: 2331683"
    assert _row(report, "PE Ratio") == "PE Ratio (fiscal year to 2024-01-28): 78.3x"
    assert _row(report, "Price to Book") == "Price to Book (equity at 2024-01-28): 54.3x"
    assert "Point-in-time as of: 2024-05-15" in report


@pytest.mark.unit
def test_a_split_between_the_count_and_the_date_restates_the_count(filings):
    """NVDA split 10:1 on 2024-06-10: the count filed in May is on the old basis."""
    filings(_facts(cover=COVERS), close=128.44, session="2024-07-15", splits={"2024-06-10": 10.0})
    report = sec_edgar.get_fundamentals("NVDA", "2024-07-15")
    assert _row(report, "Shares Outstanding") == (
        "Shares Outstanding (millions): 24600.0 (counted 2024-05-24, restated for the split since)")
    assert _row(report, "Market Cap") == "Market Cap: 3159624"


@pytest.mark.unit
def test_a_close_from_before_a_later_split_is_put_back_on_its_own_basis(monkeypatch):
    """Yahoo serves 94.63 for 2024-05-15 after the 10:1 split; 946.30 is what traded."""
    class Handle:
        splits = pd.Series([10.0], index=pd.to_datetime(["2024-06-10"]))

        def history(self, **_):
            return pd.DataFrame({"Close": [94.63]}, index=pd.to_datetime(["2024-05-15"]))

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: Handle())
    close, session, splits = sec_edgar._close("NVDA", "2024-05-15")
    assert (round(close, 2), session) == (946.30, "2024-05-15")


@pytest.mark.unit
def test_a_twelve_month_10q_is_not_taken_for_the_fiscal_year(filings):
    filings(_facts(cover=COVERS))
    report = sec_edgar.get_fundamentals("NVDA", "2024-07-15")
    assert "fiscal year to 2024-01-28" in _row(report, "PE Ratio")


@pytest.mark.unit
def test_a_loss_makes_the_pe_not_meaningful(filings):
    filings(_facts(cover=COVERS, net_income=-500_000_000))
    report = sec_edgar.get_fundamentals("NVDA", "2024-05-15")
    assert _row(report, "PE Ratio").endswith("not meaningful (zero or negative denominator)")


@pytest.mark.unit
def test_without_a_cover_page_total_the_balance_sheet_count_stands_in(filings):
    """Filers with several share classes file no single cover-page total."""
    filings(_facts(balance_count=_fact("2024-01-28", 2_464_000_000, "2024-02-21")))
    report = sec_edgar.get_fundamentals("NVDA", "2024-05-15")
    assert "(counted 2024-01-28)" in _row(report, "Shares Outstanding")


@pytest.mark.unit
def test_a_stale_share_count_gives_no_valuation(filings):
    filings(_facts(cover=[_fact("2022-02-16", 2_500_000_000, "2022-02-21")]))
    with pytest.raises(NoMarketDataError, match="no share count filed near"):
        sec_edgar.get_fundamentals("NVDA", "2024-05-15")


@pytest.mark.unit
def test_today_is_left_to_the_next_vendor(filings, monkeypatch):
    """A run dated today keeps the live profile, with its sector and forward estimates."""
    filings(_facts(cover=COVERS))
    with pytest.raises(NoMarketDataError, match="past dates only"):
        sec_edgar.get_fundamentals("NVDA", None)
    monkeypatch.setitem(route_to_vendor.__globals__["VENDOR_METHODS"]["get_fundamentals"],
                        "yfinance", lambda ticker, as_of_date=None: "live profile")
    assert route_to_vendor("get_fundamentals", "NVDA", None) == "live profile"
