from app.risk.position_sizing import should_rebalance


def test_no_change_never_rebalances():
    assert should_rebalance(10, 10) is False


def test_opening_from_flat_always_executes():
    assert should_rebalance(0, 5) is True


def test_closing_to_flat_always_executes():
    assert should_rebalance(5, 0) is True


def test_direction_flip_always_executes():
    assert should_rebalance(5, -3) is True


def test_small_same_direction_wobble_is_ignored():
    # 10 -> 11 is a 10% change, below the 20% default threshold.
    assert should_rebalance(10, 11) is False


def test_meaningful_same_direction_change_executes():
    # 10 -> 13 is a 30% change, above the 20% default threshold.
    assert should_rebalance(10, 13) is True


def test_small_positions_still_get_a_minimum_one_contract_floor():
    # current=2, 20% of 2 rounds to 0 -- floor of 1 contract keeps it responsive
    # to GROWTH (only shrinks are held below min_shrink_contracts).
    assert should_rebalance(2, 3) is True


def test_shrinking_a_small_position_holds_flat_instead_of_nibbling():
    # Regression test: current=3 -> target=2 previously passed the "meaningful
    # fraction" check (20% of 3 rounds to 1, and the change here is exactly
    # 1), triggering a real 1-contract partial-close trade (and its fee) for
    # what's really just noise on an already-small position. Below
    # min_shrink_contracts, a shrink like this should hold instead.
    assert should_rebalance(3, 2) is False
    assert should_rebalance(5, 4) is False
    assert should_rebalance(1, -1) is True  # still flips direction, not a shrink


def test_shrinking_a_small_position_to_flat_still_executes():
    # target=0 hits the "closing to flat" branch before the shrink guard.
    assert should_rebalance(3, 0) is True


def test_shrinking_a_position_above_the_small_threshold_is_unaffected():
    # current=10 is above the default min_shrink_contracts=5, so the normal
    # meaningful-fraction rule still applies: 10 -> 9 is a 10% change, below
    # the 20% threshold, so it's ignored (not because it's "small", but
    # because it's not a meaningful fraction).
    assert should_rebalance(10, 9) is False
    # 10 -> 7 is a 30% change -- meaningful, executes.
    assert should_rebalance(10, 7) is True


def test_growing_a_small_position_is_unaffected_by_the_shrink_guard():
    assert should_rebalance(3, 4) is True
