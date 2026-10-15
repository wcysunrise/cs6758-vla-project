import os
os.environ["MPLBACKEND"]="Agg"
#!/usr/bin/env python3
"""Download pinned inference assets and inspect the selected task (no training)."""
import json
from run_baseline import ROOT, load_config, resolve_snapshots, configure_libero

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs/baseline.yaml")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    snapshots, lock = resolve_snapshots(cfg, out / "hub_revisions.json")
    configure_libero(snapshots["assets"])
    from libero.libero import benchmark
    task = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0).get_task(0)
    print(json.dumps({"snapshots": {k: str(v) for k, v in snapshots.items()},
                      "task_name": task.name, "instruction": task.language}, indent=2))
