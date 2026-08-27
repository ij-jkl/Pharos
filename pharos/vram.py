"""What a context token actually costs in VRAM on THIS machine, measured rather than assumed.

`pharos.profiler.backend._extract_kv_bytes_per_token` derives the KV rate from the model's own
GGUF metadata, and refuses to for two architecture families: hybrid SSM stacks that keep a cache
on only some layers, and sliding-window attention where *which* layers are windowed is not
published. Those fall back to the configured `kv_mib_per_1k` -- one constant for every model,
which measured 4-7x wrong across three architectures, in the direction that overstates how much
context still fits.

There was never any need to guess. `/api/ps` reports `size_vram` and the loaded window on every
probe, VRAM is linear in context, and Pharos probes the backend constantly -- so loading a model
at several windows over time hands the true rate over as the slope of a line it can fit itself.
That is exactly how the numbers in `_extract_kv_bytes_per_token`'s validation table were
obtained; this module does it without anyone running curl by hand.

The methodology is that table's, unchanged, because it is the one that was validated:

* **Fit from 8K upward.** Below roughly 8K the line bends -- a fixed allocation is still
  changing size -- and a slope fitted through that region is not the KV rate.
* **At least three points**, so a single odd reading cannot be the answer.
* **Only a fully resident model.** `size_vram == size` is Ollama's own statement that nothing
  was offloaded to RAM; a partly offloaded model's VRAM does not move with context in any way
  worth fitting, and mixing the two shapes gives a slope that describes neither.
* **One digest.** A model re-pulled under the same tag can be a different file with a different
  cache, so a changed digest discards what was learned about the old one.

Fail any of those and there is no measurement, which is reported as such rather than filled in.
The store is counts only -- a window size and a byte total per model -- it is gitignored, and
deleting it costs nothing but the re-measuring.

One thing the first live fit turned up, recorded here because it qualifies every number this
module produces. `qwen3.5:9b` on an RTX 3060 reports `size_vram` that is exactly linear inside a
regime and then changes slope:

    ======  ======  =================
    from      to     MiB per 1K
    ======  ======  =================
     8,192  16,384   33.20
    16,384  32,768   33.20
    32,768  65,536   28.56
    ======  ======  =================

Two intervals at 34,816 bytes per token to the byte, and then a third at 29,952. So the curve
is piecewise linear, not linear, and a least-squares line through all four points describes
neither piece exactly -- it read 30 where the top segment is 28.6 and the bottom two are 33.2.

The fit is left as a straight line through everything above the floor anyway, for two reasons.
The bend is downward, because a fixed allocation is being amortised over more tokens, which is
the same reason the line bends below 8K; and a rate fitted too HIGH spends the headroom estimate
too fast, which is the direction this number is allowed to be wrong in. Fitting only the top
segment would answer the marginal question more exactly and would do it from two points, which
is the shortcut the methodology above exists to refuse. The span is reported alongside the rate
so the windows behind it are visible rather than implied.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

_BYTES_PER_MIB = 1024 * 1024

# Below this window the VRAM/context line bends, so points under it are recorded but never
# fitted. See the validation table in `_extract_kv_bytes_per_token`.
FIT_FLOOR_CTX = 8192
# One reading is an anecdote and two cannot outvote a bad one.
MIN_FIT_POINTS = 3
# The two windows fitted across have to be far enough apart that noise in `size_vram` does not
# dominate the slope. One doubling from the floor.
MIN_FIT_SPAN = 8192
# A rate outside this band is not a KV cache; something else moved between the two readings.
# The widest real figure measured here is 142.6 MiB/1K (qwen3-4b), the narrowest 32.2.
_PLAUSIBLE_MIB_PER_1K = (1.0, 1000.0)
# Windows kept per model. Generous: each is two numbers, and a long history costs nothing.
_MAX_WINDOWS = 32

_logger = logging.getLogger("pharos.vram")


@dataclass(frozen=True, slots=True)
class MeasuredKv:
    """A KV rate fitted from this machine's own `/api/ps` readings."""

    model: str
    mib_per_1k: float
    points: int  # windows the fit used
    low_ctx: int
    high_ctx: int

    @property
    def span(self) -> str:
        """The windows fitted across, for a report that has to say where a number came from."""
        return f"{self.low_ctx:,}-{self.high_ctx:,}"


def record_vram_sample(
    path: Path,
    model: str | None,
    *,
    digest: str | None,
    loaded_ctx: int | None,
    size_vram_bytes: int | None,
    weight_size_bytes: int | None,
) -> None:
    """Remember what this model occupied at this window. Never raises.

    Same rule as every other store here: it makes the report better when it is there and must
    never be able to stop one happening.
    """
    if not model or not loaded_ctx or not size_vram_bytes or loaded_ctx <= 0:
        return
    # Ollama's own statement that nothing was offloaded. Unknown is not good enough: a sample
    # that might be half in RAM would sit in the store indistinguishable from a clean one.
    if weight_size_bytes is None or size_vram_bytes != weight_size_bytes:
        return
    try:
        store = _read(path)
        entry = store.get(model)
        if entry is None or entry.get("digest") != digest:
            entry = {"digest": digest, "windows": {}}
        windows = entry.get("windows")
        if not isinstance(windows, dict):
            windows = {}
        windows[str(loaded_ctx)] = size_vram_bytes
        if len(windows) > _MAX_WINDOWS:
            # Keep the largest windows: the fit only ever uses points above the floor.
            keep = sorted(windows, key=lambda key: int(key))[-_MAX_WINDOWS:]
            windows = {key: windows[key] for key in keep}
        entry["windows"] = windows
        entry["updated"] = time.time()
        store[model] = entry
        _write(path, store)
    except (OSError, ValueError, TypeError) as exc:
        _logger.warning("could not record a vram sample in %s: %s", path, exc)


def measured_rate(path: Path, model: str | None, *, digest: str | None = None) -> MeasuredKv | None:
    """The KV rate fitted from this machine's readings, or None when there is not enough.

    None is the honest answer to "how much does a context token cost here" until three windows
    above the fit floor have actually been seen. Everything about the shape of the fit is in
    this module's docstring; nothing here fills a gap with a plausible number.
    """
    if not model:
        return None
    try:
        entry = _read(path).get(model)
    except (OSError, ValueError, TypeError) as exc:
        _logger.warning("could not read vram samples from %s: %s", path, exc)
        return None
    if not isinstance(entry, dict):
        return None
    if digest is not None and entry.get("digest") != digest:
        return None  # a different file under the same tag

    points = _points(entry.get("windows"))
    if len(points) < MIN_FIT_POINTS:
        return None
    low_ctx, high_ctx = points[0][0], points[-1][0]
    if high_ctx - low_ctx < MIN_FIT_SPAN:
        return None

    slope = _slope(points)
    if slope is None or slope <= 0:
        return None
    mib_per_1k = slope * 1000 / _BYTES_PER_MIB
    floor, ceiling = _PLAUSIBLE_MIB_PER_1K
    if not floor <= mib_per_1k <= ceiling:
        _logger.warning("discarding an implausible vram fit for %s: %.1f MiB/1K", model, mib_per_1k)
        return None
    return MeasuredKv(
        model=model,
        mib_per_1k=mib_per_1k,
        points=len(points),
        low_ctx=low_ctx,
        high_ctx=high_ctx,
    )


def _points(windows: object) -> list[tuple[int, int]]:
    """(ctx, vram_bytes) above the fit floor, ascending. Anything malformed is dropped."""
    if not isinstance(windows, dict):
        return []
    out: list[tuple[int, int]] = []
    for key, value in windows.items():
        try:
            ctx, vram = int(key), int(value)
        except (TypeError, ValueError):
            continue
        if ctx >= FIT_FLOOR_CTX and vram > 0:
            out.append((ctx, vram))
    return sorted(out)


def _slope(points: list[tuple[int, int]]) -> float | None:
    """Least-squares slope in bytes per context token.

    Least squares rather than the widest pair: every point is a real reading, and one taken
    while something else on the card was moving should be outvoted rather than trusted for
    being at an extreme.
    """
    n = len(points)
    mean_ctx = sum(ctx for ctx, _ in points) / n
    mean_vram = sum(vram for _, vram in points) / n
    variance = sum((ctx - mean_ctx) ** 2 for ctx, _ in points)
    if variance <= 0:
        return None
    covariance = sum((ctx - mean_ctx) * (vram - mean_vram) for ctx, vram in points)
    return covariance / variance


def _read(path: Path) -> dict[str, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        return {}
    models = raw.get("models")
    if not isinstance(models, dict):
        return {}
    return {name: item for name, item in models.items() if isinstance(item, dict)}


def _write(path: Path, store: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": store}, indent=2), encoding="utf-8")
