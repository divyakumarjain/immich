"""Time one .hef under each way of driving it: device batch size, raw uint8 input, host-side dequantization.

Every configuration runs in a fresh process, so none inherits another's configured network group, and each also
returns its outputs on one fixed input so a faster path that changes the numbers is caught.

    python bench.py arcface_r50.hef --frames 1 5 16 --batch 1 8 16
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np


def worker(path: str, batch: int, raw_in: bool, raw_out: bool, frames: list[int], repeat: int, queue: Any) -> None:
    from hef_runner import HefModel

    model = HefModel(Path(path), batch_size=batch, raw_input=raw_in, raw_output=raw_out)
    node = model.inputs[0]
    rng = np.random.default_rng(0)
    timings = {}
    for n in frames:
        x = rng.integers(0, 256, (n, *node.shape), dtype=np.uint8)
        for _ in range(3):
            model.run({node.name: x})
        start = time.perf_counter()
        for _ in range(repeat):
            model.run({node.name: x})
        call = (time.perf_counter() - start) * 1000 / repeat
        timings[n] = {"ms_per_call": round(call, 2), "ms_per_frame": round(call / n, 2)}
    probe = model.run({node.name: rng.integers(0, 256, (1, *node.shape), dtype=np.uint8)})
    queue.put({"multi_context": model.contexts, "timings": timings, "probe": {k: v.ravel() for k, v in probe.items()}})
    model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hef", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 5, 16])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    baseline: dict[str, np.ndarray] | None = None
    rows = []
    for batch, raw_in, raw_out in itertools.product(args.batch, (False, True), (False, True)):
        queue = ctx.Queue()
        proc = ctx.Process(target=worker, args=(str(args.hef), batch, raw_in, raw_out, args.frames, args.repeat, queue))
        proc.start()
        result = queue.get()
        proc.join()
        baseline = baseline or result["probe"]
        drift = max(float(np.abs(result["probe"][k] - baseline[k]).max()) for k in baseline)
        label = f"batch={batch:<2} in={'uint8' if raw_in else 'f32  '} out={'host' if raw_out else 'f32 '}"
        cells = "  ".join(
            f"{n}f: {t['ms_per_call']:7.2f}ms ({t['ms_per_frame']:6.2f}/f)" for n, t in result["timings"].items()
        )
        print(f"{label}  {cells}  drift={drift:.3g}", flush=True)
        rows.append({"batch": batch, "raw_input": raw_in, "raw_output": raw_out, "drift": drift, **result["timings"]})
    print(json.dumps({"hef": args.hef.name, "multi_context": result["multi_context"], "rows": rows}))


if __name__ == "__main__":
    main()
