"""SEC EDGAR valuation: the multiples a past date could have seen.

Vendor profiles carry today's market cap and ratios, so a historical run gets a
withheld notice (#1300) and reasons about statements with no price attached.
EDGAR dates every figure by its filing, which is enough to rebuild the ratios as
they stood: the share count and trailing figures on file by the date, and that
day's close.
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows import interface, sec_edgar
from tradingagents.dataflows.errors import NoMarketDataError

_REAL_CLOSE = sec_edgar._close_and_splits

TICKER_MAP = {"0": {"cik_str": 1, "ticker": "ACME", "title": "Acme Corp"}}


def _fact(end, val, filed, start=None, form="10-Q"):
    fact = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(end[:4]), "fp": "FY"}
    if start:
        fact["start"] = start
    return fact


def _usd(*facts):
    return {"units": {"USD": list(facts)}}


# A calendar-year filer: FY2024 filed in February 2025, the half year in August
# 2025, FY2025 in February 2026. Net income: 100 for 2024, 30 then 40 for the two
# first halves, so the trailing year at mid 2025 is 100 + 40 - 30 = 110.
FACTS = {"facts": {
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        _fact("2025-01-31", 10_000_000, "2025-02-10", form="10-K"),
        _fact("2025-07-25", 10_000_000, "2025-08-01"),
        _fact("2026-01-30", 10_000_000, "2026-02-10", form="10-K"),
    ]}}},
    "us-gaap": {
        "NetIncomeLoss": _usd(
            _fact("2024-12-31", 100e6, "2025-02-10", start="2024-01-01", form="10-K"),
            _fact("2024-06-30", 30e6, "2024-08-01", start="2024-01-01"),
            _fact("2025-06-30", 40e6, "2025-08-01", start="2025-01-01"),
            _fact("2025-06-30", 25e6, "2025-08-01", start="2025-04-01"),
            _fact("2025-12-31", 50e6, "2026-02-10", start="2025-01-01", form="10-K"),
        ),
        "Revenues": _usd(
            _fact("2024-12-31", 900e6, "2025-02-10", start="2024-01-01", form="10-K"),
            _fact("2024-06-30", 400e6, "2024-08-01", start="2024-01-01"),
            _fact("2025-06-30", 500e6, "2025-08-01", start="2025-01-01"),
        ),
        "Assets": _usd(_fact("2024-12-31", 800e6, "2025-02-10", form="10-K"),
                       _fact("2025-06-30", 850e6, "2025-08-01")),
        "StockholdersEquity": _usd(_fact("2025-06-30", 500e6, "2025-08-01")),
        "CashAndCashEquivalentsAtCarryingValue": _usd(_fact("2024-12-31", 90e6, "2025-02-10"),
                                                      _fact("2025-06-30", 60e6, "2025-08-01")),
        "LongTermDebtNoncurrent": _usd(_fact("2025-06-30", 200e6, "2025-08-01")),
    },
}}


@pytest.fixture(autouse=True)
def _no_network_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sec_edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else FACTS)
    monkeypatch.setattr(sec_edgar, "_close_and_splits",
                        lambda ticker, curr_date: (110.0, curr_date, {}))


def _row(report: str, label: str) -> str:
    return [line for line in report.splitlines() if line.startswith(label)][0]


@pytest.mark.unit
def test_the_trailing_year_is_built_from_three_filed_figures():
    """Fiscal year, plus the year to date, minus the same stretch a year before."""
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    row = _row(out, "Net Income (TTM)")
    assert row.startswith("Net Income (TTM): 110 ")
    # The parts are named, so the sum can be checked against the filings.
    assert "2024-12-31 (100)" in row and "2025-06-30 (40)" in row and "2024-06-30 (30)" in row


@pytest.mark.unit
def test_the_multiples_use_that_day_s_close_and_the_count_on_file():
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert _row(out, "Market Cap") == "Market Cap: 1100"          # 110.00 x 10M shares
    assert _row(out, "PE Ratio (TTM)") == "PE Ratio (TTM): 10.0x"  # 1100 / 110
    assert _row(out, "Price to Sales (TTM)") == "Price to Sales (TTM): 1.1x"  # 1100 / 1000
    assert _row(out, "Price to Book") == "Price to Book: 2.2x"    # 1100 / 500


@pytest.mark.unit
def test_a_half_year_that_was_not_filed_yet_is_not_in_the_trailing_year():
    """The half ended 2025-06-30 and reached the public on 2025-08-01."""
    out = sec_edgar.get_fundamentals("ACME", "2025-07-15")
    assert _row(out, "Net Income (TTM)") == "Net Income (TTM): 100 = fiscal year to 2024-12-31"
    assert "2025-06-30" not in out


@pytest.mark.unit
def test_right_after_the_annual_report_the_fiscal_year_is_the_trailing_year():
    out = sec_edgar.get_fundamentals("ACME", "2026-03-01")
    assert _row(out, "Net Income (TTM)") == "Net Income (TTM): 50 = fiscal year to 2025-12-31"


@pytest.mark.unit
def test_a_quarter_reported_alone_is_never_added_as_if_it_were_a_half_year():
    """A second quarter with no six-month figure cannot extend a fiscal year."""
    facts = {"NetIncomeLoss": _usd(
        _fact("2024-12-31", 100e6, "2025-02-10", start="2024-01-01", form="10-K"),
        _fact("2024-06-30", 20e6, "2024-08-01", start="2024-04-01"),
        _fact("2025-06-30", 25e6, "2025-08-01", start="2025-04-01"),
    )}
    trailing = sec_edgar._trailing_year(facts, ("NetIncomeLoss",), "2025-09-15")
    assert trailing == {"value": 100e6, "end": "2024-12-31", "parts": "fiscal year to 2024-12-31"}


@pytest.mark.unit
def test_a_split_after_the_count_restates_the_count(monkeypatch):
    """Shares were counted before a 10-for-1 split; the close is after it."""
    monkeypatch.setattr(sec_edgar, "_close_and_splits",
                        lambda ticker, curr_date: (11.0, curr_date, {"2025-08-20": 10.0}))
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert "Shares Outstanding (millions): 100.0" in out and "restated for the split" in out
    assert _row(out, "Market Cap") == "Market Cap: 1100"


@pytest.mark.unit
def test_a_later_split_is_undone_so_the_price_is_the_one_that_traded(monkeypatch):
    """Yahoo restates past closes for later splits: 110 traded, 11 is served today."""
    import pandas as pd

    class _Ticker:
        splits = pd.Series([10.0], index=pd.to_datetime(["2026-01-15"]))

        def history(self, **kwargs):
            return pd.DataFrame({"Close": [11.0]}, index=pd.to_datetime(["2025-09-15"]))

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: _Ticker())
    close, session, splits = _REAL_CLOSE("ACME", "2025-09-15")
    assert (round(close, 2), session) == (110.0, "2025-09-15")
    assert splits == {"2026-01-15": 10.0}


@pytest.mark.unit
def test_a_loss_has_no_price_to_earnings(monkeypatch):
    import copy

    facts = copy.deepcopy(FACTS)
    for fact in facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]:
        fact["val"] = -abs(fact["val"])
    monkeypatch.setattr(sec_edgar, "_fetch_json",
                        lambda url: TICKER_MAP if "company_tickers" in url else facts)
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert "not meaningful" in _row(out, "PE Ratio (TTM)")


@pytest.mark.unit
def test_an_input_that_was_never_filed_is_named_unavailable_not_guessed():
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert "unavailable" in _row(out, "Free Cash Flow Yield (TTM)")   # no cash flow tagged
    assert _row(out, "PE Ratio (TTM)") == "PE Ratio (TTM): 10.0x"  # the rest still returns



@pytest.mark.unit
def test_a_company_that_stopped_filing_gets_no_multiples():
    with pytest.raises(NoMarketDataError, match="no share count filed near"):
        sec_edgar.get_fundamentals("ACME", "2028-01-01")


@pytest.mark.unit
def test_the_report_states_the_vintage_rule_and_what_it_leaves_out():
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert "filed on or before 2025-09-15" in out
    assert "forward estimates have no filed vintage" in out
    assert "Acme" not in out  # a name is today's name


@pytest.mark.unit
def test_a_non_filer_falls_through_to_the_next_vendor():
    with pytest.raises(NoMarketDataError, match="not a US SEC filer"):
        sec_edgar.get_fundamentals("0700.HK", "2025-09-15")


@pytest.mark.unit
def test_the_vendor_is_routable_for_fundamentals():
    assert interface.VENDOR_METHODS["get_fundamentals"]["sec_edgar"] is sec_edgar.get_fundamentals


@pytest.mark.unit
def test_book_value_older_than_the_stale_guard_is_not_used():
    """Equity is on file only at mid 2025; a year later it no longer prices the book."""
    out = sec_edgar.get_fundamentals("ACME", "2026-03-01")
    assert "unavailable" in _row(out, "Price to Book")


@pytest.mark.unit
def test_enterprise_value_is_deliberately_not_served():
    out = sec_edgar.get_fundamentals("ACME", "2025-09-15")
    assert "Enterprise Value" not in out and "EBITDA" not in out
