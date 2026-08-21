from app.data.market_cache import MarketDataCache


def test_smooth_imbalance_first_reading_returns_itself():
    cache = MarketDataCache()
    assert cache.smooth_imbalance("BTCUSD", "book", 0.8) == 0.8


def test_smooth_imbalance_damps_a_sign_flip_instead_of_snapping_to_it():
    """Regression test: raw book/trade-flow imbalance used to be fed straight
    into the live signal every tick with no memory of prior ticks (the
    feature frame it lived in was rebuilt from scratch each time), so a
    single order-book snapshot flipping sign flipped the strategy's position
    immediately. A persistent EMA should land somewhere BETWEEN the old and
    new reading on a flip, not jump straight to the new value.
    """
    cache = MarketDataCache()
    cache.smooth_imbalance("BTCUSD", "book", 0.8)
    flipped = cache.smooth_imbalance("BTCUSD", "book", -0.8)

    # Damped, not snapped: still short of the new extreme in one tick.
    assert -0.8 < flipped < 0.8
    # A genuine, SUSTAINED flip should still eventually show up (this isn't
    # a permanent lock at the old value).
    second = cache.smooth_imbalance("BTCUSD", "book", -0.8)
    assert second < flipped


def test_smooth_imbalance_converges_toward_a_sustained_new_reading():
    cache = MarketDataCache()
    cache.smooth_imbalance("BTCUSD", "book", 0.0)
    last = 0.0
    for _ in range(30):
        last = cache.smooth_imbalance("BTCUSD", "book", 1.0)
    assert last > 0.95  # converges close to the sustained value


def test_smooth_imbalance_is_isolated_per_symbol_and_per_signal_key():
    cache = MarketDataCache()
    cache.smooth_imbalance("BTCUSD", "book", 0.5)
    cache.smooth_imbalance("BTCUSD", "flow", -0.5)
    cache.smooth_imbalance("ETHUSD", "book", 0.9)

    assert cache.smooth_imbalance("BTCUSD", "book", None) == 0.5
    assert cache.smooth_imbalance("BTCUSD", "flow", None) == -0.5
    assert cache.smooth_imbalance("ETHUSD", "book", None) == 0.9


def test_smooth_imbalance_holds_last_value_when_reading_is_missing():
    cache = MarketDataCache()
    cache.smooth_imbalance("BTCUSD", "book", 0.4)
    assert cache.smooth_imbalance("BTCUSD", "book", None) == 0.4


def test_smooth_imbalance_returns_none_when_never_seen():
    cache = MarketDataCache()
    assert cache.smooth_imbalance("BTCUSD", "book", None) is None
