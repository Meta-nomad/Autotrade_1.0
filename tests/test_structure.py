from app.models import MinuteBar, Side
from app.structure import analyse_structure, confirmed_pivots


def _bar(index: int, close: float) -> MinuteBar:
    return MinuteBar(
        ts=index * 14_400,
        open=close - 0.1,
        high=close + 0.4,
        low=close - 0.4,
        close=close,
        volume_notional=1_000,
    )


def test_confirmed_pivots_never_use_unconfirmed_tail() -> None:
    bars = [_bar(index, value) for index, value in enumerate((1, 2, 4, 2, 1, 3, 9))]

    highs, _ = confirmed_pivots(bars, left=2, right=2)

    assert highs == [(2, 4.4)]
    assert all(index < len(bars) - 2 for index, _ in highs)


def test_structure_detects_breakout_without_future_bars() -> None:
    import math

    bars = [
        _bar(index, 100 + index * 0.42 + math.sin(index * math.pi / 3) * 2.2)
        for index in range(45)
    ]
    bars[-1] = _bar(44, bars[-1].close + 7.0)
    bars[-1].volume_notional = 2_000

    context = analyse_structure(bars, Side.LONG)

    assert context.ready
    assert context.breakout
    assert context.volume_ratio > 1.5
