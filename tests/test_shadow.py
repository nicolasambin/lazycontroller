import pytest

from lazycontroller import Route, ShadowReport


def shadow(confidence, agree=True):
    return Route(
        "chat.completions",
        "slow",
        "shadow",
        "gpt-6-luna",
        confidence=confidence,
        shadow_mismatches=() if agree else ("category",),
    )


# Confident calls all agree; below 0.7 half of them don't.
ROUTES = [
    *(shadow(c) for c in (1.0, 0.99, 0.98, 0.95, 0.91, 0.88, 0.85, 0.81)),
    shadow(0.76, agree=False),
    *(shadow(c) for c in (0.72, 0.7)),
    shadow(0.65, agree=False),
    shadow(0.6),
    shadow(0.45, agree=False),
    shadow(0.4),
]


def test_ignores_routes_that_are_not_from_shadow_mode():
    other = Route("chat.completions", "fast", "confident", "gpt-6-luna", confidence=0.99)
    ineligible = Route("chat.completions", "slow", "ineligible", "gpt-6-luna")
    assert len(ShadowReport([*ROUTES, other, ineligible])) == len(ROUTES)


def test_at_counts_what_the_fast_model_would_take():
    point = ShadowReport(ROUTES).at(0.8)
    assert (point.taken, point.agreed, point.total) == (8, 8, 15)
    assert point.coverage == pytest.approx(8 / 15)
    assert point.agreement == 1.0
    assert ShadowReport(ROUTES).at(0.7).agreement == pytest.approx(10 / 11)


def test_agreement_is_none_when_nothing_is_taken():
    assert ShadowReport(ROUTES).at(1.01).agreement is None
    assert ShadowReport([]).at(0.5).coverage == 0.0


def test_curve_defaults_to_a_grid_from_0_to_1():
    curve = ShadowReport(ROUTES).curve()
    assert [p.min_confidence for p in curve][:3] == [0.0, 0.05, 0.1]
    assert curve[-1].min_confidence == 1.0
    assert curve[0].taken == len(ROUTES)


def test_bands_group_by_confidence_highest_first():
    bands = ShadowReport(ROUTES).bands()
    assert [(b.low, b.high, b.count, b.agreed) for b in bands] == [
        (0.9, 1.0, 5, 5),
        (0.7, 0.9, 6, 5),
        (0.0, 0.7, 4, 2),
    ]


def test_min_confidence_is_the_lowest_one_that_keeps_agreement():
    report = ShadowReport(ROUTES)
    assert report.min_confidence(1.0, min_samples=5) == 0.81
    # 0.7 alone would give 10/11, but 0.76 above it gives 8/9 (89%): not good enough for 0.9.
    assert report.min_confidence(0.9, min_samples=5) == 0.81
    assert report.min_confidence(0.85, min_samples=5) == 0.7
    assert report.min_confidence(0.0, min_samples=5) == 0.4


def test_min_confidence_needs_enough_samples():
    report = ShadowReport(ROUTES)
    assert report.min_confidence(1.0) is None  # default min_samples=20, only 15 calls
    assert report.min_confidence(1.0, min_samples=9) is None  # 0.81 covers only 8


def test_min_confidence_does_not_rest_on_a_lucky_dip():
    # Agreement is 100% at 0.5 only because a disagreement drops out between 0.5 and 0.6.
    routes = [shadow(0.9), shadow(0.8), shadow(0.7, agree=False), shadow(0.6), shadow(0.5)]
    assert ShadowReport(routes).min_confidence(0.8, min_samples=1) == 0.8


def test_min_confidence_validates_min_agreement():
    with pytest.raises(ValueError):
        ShadowReport(ROUTES).min_confidence(1.5)


def test_table_lists_bands_and_min_confidences():
    table = ShadowReport(ROUTES).table()
    assert table.splitlines()[0] == "15 shadow calls"
    assert "0.90-1.00           5  5/5 (100%)" in table
    assert "0.80           8/15 (53%)      8/8 (100%)" in table
