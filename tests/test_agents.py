from unittest.mock import MagicMock, patch

import pytest

from src.agents.ben_graham import ben_graham_agent
from src.graph.state import AgentState


class MockData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


@pytest.fixture
def mock_state():
    return {
        "messages": [],
        "data": {
            "tickers": ["AAPL"],
            "end_date": "2024-01-01",
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": "gpt-4o",
            "model_provider": "OpenAI",
        },
    }


@patch("src.agents.ben_graham.get_financial_metrics")
@patch("src.agents.ben_graham.search_line_items")
@patch("src.agents.ben_graham.get_market_cap")
@patch("src.agents.ben_graham.call_llm")
@patch("src.agents.ben_graham.progress")
def test_ben_graham_agent_bullish(
    mock_progress, mock_call_llm, mock_get_market_cap, mock_search_line_items, mock_get_financial_metrics, mock_state
):
    # Setup mocks
    mock_get_financial_metrics.return_value = [MockData()]

    mock_item = MockData(
        earnings_per_share=10.0,
        total_assets=1000.0,
        total_liabilities=200.0,
        current_assets=500.0,
        current_liabilities=100.0,
        book_value_per_share=50.0,
        outstanding_shares=100.0,
        dividends_and_other_cash_distributions=-1.0,
    )
    mock_search_line_items.return_value = [mock_item, mock_item]  # Two items for growth check

    mock_get_market_cap.return_value = 500.0

    mock_output = MagicMock()
    mock_output.signal = "bullish"
    mock_output.confidence = 90.0
    mock_output.reasoning = "Strong value play"
    mock_call_llm.return_value = mock_output

    # Run agent
    result = ben_graham_agent(mock_state)

    # Assertions
    assert "ben_graham_agent" in result["data"]["analyst_signals"]
    assert result["data"]["analyst_signals"]["ben_graham_agent"]["AAPL"]["signal"] == "bullish"
    assert result["data"]["analyst_signals"]["ben_graham_agent"]["AAPL"]["confidence"] == 90.0
    mock_call_llm.assert_called_once()


@patch("src.agents.warren_buffett.get_financial_metrics")
@patch("src.agents.warren_buffett.search_line_items")
@patch("src.agents.warren_buffett.get_market_cap")
@patch("src.agents.warren_buffett.call_llm")
@patch("src.agents.warren_buffett.progress")
def test_warren_buffett_agent(
    mock_progress, mock_call_llm, mock_get_market_cap, mock_search_line_items, mock_get_financial_metrics, mock_state
):
    from src.agents.warren_buffett import warren_buffett_agent

    # Setup mocks
    mock_metric = MockData(
        return_on_equity=0.20,
        debt_to_equity=0.3,
        operating_margin=0.25,
        current_ratio=2.0,
        asset_turnover=1.2,
        return_on_invested_capital=0.18,
    )
    mock_get_financial_metrics.return_value = [mock_metric] * 10

    mock_item = MockData(
        net_income=100.0,
        depreciation_and_amortization=20.0,
        capital_expenditure=-10.0,
        outstanding_shares=100.0,
        shareholders_equity=500.0,
        revenue=1000.0,
        issuance_or_purchase_of_equity_shares=-50.0,
        dividends_and_other_cash_distributions=-20.0,
        gross_profit=500.0,
        gross_margin=0.5,
    )
    mock_search_line_items.return_value = [mock_item] * 10

    mock_get_market_cap.return_value = 1000.0

    mock_output = MagicMock()
    mock_output.signal = "bullish"
    mock_output.confidence = 85.0
    mock_output.reasoning = "Wonderful business"
    mock_call_llm.return_value = mock_output

    # Run agent
    result = warren_buffett_agent(mock_state)

    # Assertions
    assert "warren_buffett_agent" in result["data"]["analyst_signals"]
    assert result["data"]["analyst_signals"]["warren_buffett_agent"]["AAPL"]["signal"] == "bullish"
    mock_call_llm.assert_called_once()
