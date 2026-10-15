#!/usr/bin/env python3
"""Initialize/reset/render the fixed LIBERO task without loading policy weights."""
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
from run_baseline import ROOT, configure_libero, sha256, dump_json

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    from pathlib import Path
    assets = Path(snapshot_download("lerobot/libero-assets", repo_type="dataset",
                                  revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"))
    configure_libero(assets)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio
    suite = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0)
    task = suite.get_task(0)
    assert task.name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                            camera_widths=256, camera_names=["agentview", "robot0_eye_in_hand"])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    try:
        env.seed(1005)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[5])
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        imageio.imwrite(out / "simulator_check.png", obs["agentview_image"][::-1, ::-1])
        result = {"status": "passed", "task_name": task.name,
                  "instruction": task.language, "success": bool(env.check_success()),
                  "image_shapes": {k: list(obs[k].shape) for k in ["agentview_image", "robot0_eye_in_hand_image"]},
                  "action_dimension": env.env.action_dim,
                  "control_frequency_hz": env.env.control_freq,
                  "controller_types": [type(r.controller).__name__ for r in env.env.robots],
                  "bddl_sha256": sha256(bddl)}
        dump_json(out / "simulator_check.json", result)
        print(json.dumps(result, indent=2))
    finally:
        env.close()
