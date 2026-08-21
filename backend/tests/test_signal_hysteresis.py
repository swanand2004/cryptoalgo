from app.risk.position_sizing import apply_signal_hysteresis


def test_flat_position_requires_entry_threshold():
    assert apply_signal_hysteresis(0.10, current_position_sign=0, entry_threshold=0.15, exit_threshold=0.05) == 0.0
    assert apply_signal_hysteresis(0.20, current_position_sign=0, entry_threshold=0.15, exit_threshold=0.05) == 0.20
    assert apply_signal_hysteresis(-0.20, current_position_sign=0, entry_threshold=0.15, exit_threshold=0.05) == -0.20


def test_long_position_holds_through_a_dip_that_does_not_cross_exit_line():
    # Signal faded from strong to weak-but-still-positive -- should hold, not flatten.
    result = apply_signal_hysteresis(0.08, current_position_sign=1, entry_threshold=0.15, exit_threshold=0.05)
    assert result == 0.08


def test_long_position_flattens_once_below_exit_threshold():
    result = apply_signal_hysteresis(0.02, current_position_sign=1, entry_threshold=0.15, exit_threshold=0.05)
    assert result == 0.0


def test_long_position_does_not_flip_on_weak_opposite_wobble():
    # Signal dipped slightly negative but not enough to justify a full reversal.
    result = apply_signal_hysteresis(-0.08, current_position_sign=1, entry_threshold=0.15, exit_threshold=0.05)
    assert result == 0.0  # flattens, but does not go short


def test_long_position_reverses_on_genuine_opposite_conviction():
    result = apply_signal_hysteresis(-0.30, current_position_sign=1, entry_threshold=0.15, exit_threshold=0.05)
    assert result == -0.30


def test_short_position_is_symmetric():
    assert apply_signal_hysteresis(-0.08, current_position_sign=-1, entry_threshold=0.15, exit_threshold=0.05) == -0.08
    assert apply_signal_hysteresis(-0.02, current_position_sign=-1, entry_threshold=0.15, exit_threshold=0.05) == 0.0
    assert apply_signal_hysteresis(0.30, current_position_sign=-1, entry_threshold=0.15, exit_threshold=0.05) == 0.30
    assert apply_signal_hysteresis(0.08, current_position_sign=-1, entry_threshold=0.15, exit_threshold=0.05) == 0.0


def test_noise_oscillating_around_entry_threshold_does_not_flip_flop_once_in_position():
    """The exact overtrading pattern this was built to fix: signal wobbling
    just above/below the entry line while already in a position should NOT
    cause repeated close/reopen cycles.
    """
    current_sign = 0
    noisy_signal = [0.16, 0.14, 0.17, 0.13, 0.16, 0.14]
    actions = []
    for raw in noisy_signal:
        effective = apply_signal_hysteresis(raw, current_sign, entry_threshold=0.15, exit_threshold=0.05)
        if effective != 0 and current_sign == 0:
            actions.append("enter")
            current_sign = 1 if effective > 0 else -1
        elif effective == 0 and current_sign != 0:
            actions.append("exit")
            current_sign = 0
        else:
            actions.append("hold")

    assert actions == ["enter", "hold", "hold", "hold", "hold", "hold"]
