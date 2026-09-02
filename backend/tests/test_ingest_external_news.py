import pytest

from app.data.ingest_external import (
    NEWS_SYMBOL_KEYWORDS,
    aggregate_news_sentiment,
    score_headline_sentiment,
)


def test_score_headline_sentiment_separates_positive_from_negative():
    headlines = [
        {"time": 1_700_000_000, "title": "Bitcoin surges to new all-time high, analysts thrilled", "source": "x"},
        {"time": 1_700_000_100, "title": "Exchange hacked, millions of dollars stolen in devastating attack", "source": "x"},
        {"time": 1_700_000_200, "title": "Bitcoin trades sideways ahead of Fed meeting", "source": "x"},
    ]
    scored = score_headline_sentiment(headlines)
    by_title = {h["title"]: h["sentiment"] for h in scored}
    assert by_title["Bitcoin surges to new all-time high, analysts thrilled"] > 0.3
    assert by_title["Exchange hacked, millions of dollars stolen in devastating attack"] < -0.3
    assert -1.0 <= by_title["Bitcoin trades sideways ahead of Fed meeting"] <= 1.0


def test_aggregate_news_sentiment_buckets_by_hour_and_averages():
    hour0 = 1_700_000_000 - (1_700_000_000 % 3600)
    scored = [
        {"time": hour0 + 10, "title": "a", "source": "x", "sentiment": 0.5},
        {"time": hour0 + 1200, "title": "b", "source": "x", "sentiment": -0.1},
        {"time": hour0 + 3600 + 5, "title": "c", "source": "x", "sentiment": 0.8},  # next hour
    ]
    candles = aggregate_news_sentiment(scored)
    assert len(candles) == 2
    first, second = candles
    assert first["time"] == hour0
    assert first["close"] == pytest.approx((0.5 - 0.1) / 2)
    assert first["volume"] == 2.0
    assert second["time"] == hour0 + 3600
    assert second["close"] == pytest.approx(0.8)
    assert second["volume"] == 1.0


def test_aggregate_news_sentiment_keyword_filter_matches_only_relevant_headlines():
    scored = [
        {"time": 1_700_000_000, "title": "Bitcoin rallies past resistance", "source": "x", "sentiment": 0.6},
        {"time": 1_700_000_050, "title": "Solana network outage reported", "source": "x", "sentiment": -0.4},
    ]
    btc_only = aggregate_news_sentiment(scored, keywords=NEWS_SYMBOL_KEYWORDS["BTCUSD"])
    sol_only = aggregate_news_sentiment(scored, keywords=NEWS_SYMBOL_KEYWORDS["SOLUSD"])
    assert len(btc_only) == 1
    assert btc_only[0]["close"] == pytest.approx(0.6)
    assert len(sol_only) == 1
    assert sol_only[0]["close"] == pytest.approx(-0.4)


def test_aggregate_news_sentiment_returns_empty_when_nothing_matches():
    scored = [{"time": 1_700_000_000, "title": "Ethereum upgrade scheduled", "source": "x", "sentiment": 0.2}]
    assert aggregate_news_sentiment(scored, keywords=NEWS_SYMBOL_KEYWORDS["SOLUSD"]) == []
    assert aggregate_news_sentiment([], keywords=None) == []
