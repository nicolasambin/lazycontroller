"""Turn shadow-mode routes into the numbers you need to pick a threshold.

In shadow mode both models answer every call; each route records the fast model's confidence
and whether the two agreed. ``ShadowReport`` answers "if the threshold were t, how much traffic
would the fast model take, and how often would it agree with the slow one?", and suggests the
lowest threshold that keeps agreement where you want it.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .lazy import Route


@dataclass(frozen=True)
class Point:
    """What the fast model would take at one threshold."""

    threshold: float
    taken: int
    """Calls whose confidence reaches the threshold."""
    agreed: int
    """Of those, the ones where the fast and slow answers matched."""
    total: int
    """All shadow calls in the report."""

    @property
    def coverage(self) -> float:
        """Share of all calls the fast model would answer."""
        return self.taken / self.total if self.total else 0.0

    @property
    def agreement(self) -> float | None:
        """Share of the taken calls where the two models agreed; None if none are taken."""
        return self.agreed / self.taken if self.taken else None


@dataclass(frozen=True)
class Band:
    """Calls whose confidence falls in ``[low, high)``."""

    low: float
    high: float
    count: int
    agreed: int

    @property
    def agreement(self) -> float | None:
        return self.agreed / self.count if self.count else None


_GRID = tuple(round(i * 0.05, 2) for i in range(21))  # 0.00, 0.05, ..., 1.00


class ShadowReport:
    """Agreement between the fast and slow models, by confidence, from shadow-mode routes.

    Routes that aren't from shadow mode (or have no confidence) are ignored, so the list
    collected by ``on_route`` can be passed as is.
    """

    def __init__(self, routes: Iterable[Route]) -> None:
        self._calls = tuple(
            (r.confidence, not r.shadow_mismatches)
            for r in routes
            if r.reason == "shadow" and r.confidence is not None
        )

    def __len__(self) -> int:
        return len(self._calls)

    def at(self, threshold: float) -> Point:
        taken = [agreed for confidence, agreed in self._calls if confidence >= threshold]
        return Point(threshold, len(taken), sum(taken), len(self._calls))

    def curve(self, thresholds: Sequence[float] = _GRID) -> list[Point]:
        """One point per threshold, by default every 0.05 from 0 to 1. Ready to plot."""
        return [self.at(t) for t in thresholds]

    def bands(self, edges: Sequence[float] = (0.0, 0.7, 0.9, 1.0)) -> list[Band]:
        """Calls grouped by confidence, highest band first; the top band includes 1.0."""
        bands = []
        for low, high in zip(edges, edges[1:], strict=False):
            top = high >= edges[-1]
            inside = [a for c, a in self._calls if low <= c and (c <= high if top else c < high)]
            bands.append(Band(low, high, len(inside), sum(inside)))
        return bands[::-1]

    def threshold(self, min_agreement: float, *, min_samples: int = 20) -> float | None:
        """The lowest threshold at which agreement stays at or above ``min_agreement``.

        "Stays" means at that threshold and at every higher one that still covers at least
        ``min_samples`` calls, so the answer doesn't rest on a lucky dip. Candidate thresholds
        are the observed confidences. Returns None when no threshold covers ``min_samples``
        calls with enough agreement: collect more shadow traffic, or lower your bar.
        """
        if not 0.0 <= min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0 and 1")
        points = [self.at(c) for c in sorted({c for c, _ in self._calls})]
        points = [p for p in points if p.taken >= min_samples]
        best = None
        for point in reversed(points):  # from the highest threshold down
            if point.agreement < min_agreement:
                break
            best = point.threshold
        return best

    def table(self, thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> str:
        """A plain-text summary: agreement by confidence band, then by threshold."""
        lines = [f"{len(self)} shadow calls", "", "confidence band   calls  agreement"]
        for band in self.bands():
            agreement = _ratio(band.agreed, band.count)
            lines.append(f"  {band.low:.2f}-{band.high:.2f}       {band.count:>5}  {agreement}")
        lines += ["", "threshold   fast takes      agreement"]
        for p in self.curve(thresholds):
            share = f"{p.taken}/{p.total} ({p.coverage:.0%})"
            lines.append(f"  {p.threshold:.2f}      {share:<14}  {_ratio(p.agreed, p.taken)}")
        return "\n".join(lines)


def _ratio(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"
