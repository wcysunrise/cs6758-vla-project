#!/usr/bin/env python3
"""Frozen SmolVLA evaluation. No cloud resource creation; no synthetic results.

--check-config uses only the standard library. The config is JSON-format YAML.
--preflight inspects the current machine without downloading/loading a policy.
--smoke executes one real episode. A completed smoke is required for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["phase", "episode_id", "suite", "task_id", "task_name", "seed",
          "initial_state_id", "status", "success", "steps", "runtime_seconds",
          "policy_inference_calls", "policy_inference_seconds", "mean_inference_seconds",
          "termination_reason", "checkpoint_revision", "config_sha256",
          "video_file", "video_sha256"]
VERSIONS = {"lerobot": "0.5.1", "hf-libero": "0.1.4", "torch": "2.7.1",
            "torchvision": "0.22.1", "torchcodec": "0.3.0", "transformers": "5.3.0",
            "mujoco": "3.3.2", "robosuite": "1.4.0", "numpy": "2.2.6"}


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    # JSON is a YAML 1.2 subset; dependency-free validation is deliberate.
    cfg = json.loads(Path(path).read_text())
    p, e, v = cfg["policy"], cfg["environment"], cfg["evaluation"]
    if p["repo_id"] != "HuggingFaceVLA/smolvla_libero":
        raise ValueError("Stage 1 requires the specified pretrained checkpoint")
    if not re.fullmatch(r"[0-9a-f]{40}", p["revision"]):
        raise ValueError("policy.revision must be a full immutable commit SHA")
    if p["device"] != "cuda" or p["dtype"] != "float32":
        raise ValueError("This Stage 1 configuration requires CUDA FP32")
    if not 5 <= v["episodes"] <= 10:
        raise ValueError("evaluation.episodes must be between 5 and 10")
    if e["suite"] != "libero_object" or e["task_id"] != 0 or e["task_order_index"] != 0:
        raise ValueError("Stage 1 task is libero_object / task 0 / order 0")
    if e["expected_task_name"] != "pick_up_the_alphabet_soup_and_place_it_in_the_basket":
        raise ValueError("Unexpected Stage 1 task name")
    if e["control_mode"] != "relative" or e["mujoco_gl"] != "egl":
        raise ValueError("Expected relative OSC control and EGL rendering")
    for value in [p["n_action_steps"], p["num_steps"], e["image_size"], e["max_steps"],
                  v["video_fps"], v["max_wall_time_seconds"]]:
        if type(value) is not int or value <= 0:
            raise ValueError("Step counts, dimensions, fps and budget must be positive integers")
    if not 1 <= p["n_action_steps"] <= 50:
        raise ValueError("n_action_steps must fit the pretrained action chunk")
    if type(e["settle_steps"]) is not int or e["settle_steps"] < 0:
        raise ValueError("settle_steps must be a nonnegative integer")
    if type(v["seed"]) is not int or not 0 <= v["seed"] < 2**32 - 11:
        raise ValueError("seed must fit NumPy's 32-bit seed range")
    if type(v["initial_state_start"]) is not int or v["initial_state_start"] < 0:
        raise ValueError("initial_state_start must be nonnegative")
    return cfg


def config_fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def episode_plan(cfg, smoke=False):
    v = cfg["evaluation"]
    # Smoke uses an independent seed/state and never enters evaluation metrics.
    indices = [v["episodes"]] if smoke else range(v["episodes"])
    return [(i + 1, v["seed"] + i, v["initial_state_start"] + i) for i in indices]


def command_output(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def preflight(cfg):
    report = {"python": platform.python_version(), "platform": platform.system(),
              "architecture": platform.machine(), "versions": {}, "blockers": []}
    report["disk"] = dict(zip(["total_bytes", "used_bytes", "free_bytes"], shutil.disk_usage(ROOT)))
    if sys.version_info[:2] != (3, 12):
        report["blockers"].append("Use CPython 3.12")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        report["blockers"].append("Target runtime must be Linux x86_64 with NVIDIA CUDA")
    for name, expected in VERSIONS.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        report["versions"][name] = version
        if version is None or version.split("+")[0] != expected:
            report["blockers"].append(f"{name}: expected {expected}, found {version}")
    report["nvidia_smi"] = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"])
    if importlib.util.find_spec("torch"):
        try:
            import torch
            report["torch_cuda"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            report["cuda_arch_list"] = torch.cuda.get_arch_list()
            if not report["cuda_available"]:
                report["blockers"].append("torch.cuda.is_available() is false")
            else:
                report["gpu"] = torch.cuda.get_device_name(0)
                report["compute_capability"] = list(torch.cuda.get_device_capability(0))
                report["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
                report["gpu_free_memory_bytes"] = torch.cuda.mem_get_info()[0]
                # A real CUDA operation catches unsupported architecture/driver combinations.
                matrix = torch.ones((64, 64), device="cuda") @ torch.ones((64, 64), device="cuda")
                torch.cuda.synchronize()
                report["cuda_matmul_passed"] = bool(torch.all(matrix == 64).item())
                if not report["cuda_matmul_passed"]:
                    report["blockers"].append("CUDA matrix multiplication returned incorrect values")
        except Exception as exc:
            report["blockers"].append(f"CUDA import/operation failed: {exc}")
    else:
        report["blockers"].append("PyTorch unavailable; CUDA operation not tested")
    report["passed"] = not report["blockers"]
    return report


def verify_smoke(path, cfg):
    marker = json.loads(Path(path).read_text())
    if marker.get("config_sha256") != config_fingerprint(cfg):
        raise ValueError("Smoke config differs; repeat the smoke test")
    if marker.get("status") != "completed" or not marker.get("video_sha256"):
        raise ValueError("Smoke did not complete with a verified robot video")
    video = Path(path).parent / marker["video_file"]
    if not video.is_file() or sha256(video) != marker["video_sha256"]:
        raise ValueError("Smoke video missing or changed")
    return marker


def append_result(path, row):
    path = Path(path)
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("CSV schema mismatch")
            if any(r["episode_id"] == str(row["episode_id"]) for r in reader):
                raise ValueError("Duplicate episode; use a fresh output directory")
    need_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if need_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def resolve_snapshots(cfg, lock_path):
    from huggingface_hub import snapshot_download
    specs = {
        "policy": {"repo_id": cfg["policy"]["repo_id"], "repo_type": "model",
                   "revision": cfg["policy"]["revision"]},
        "backbone": {"repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "repo_type": "model",
                     "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467"},
        "assets": {"repo_id": "lerobot/libero-assets", "repo_type": "dataset",
                   "revision": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"},
    }
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        for key, spec in specs.items():
            if lock[key]["repo_id"] != spec["repo_id"] or lock[key]["repo_type"] != spec["repo_type"]:
                raise ValueError("Hub lock repo mismatch")
        if lock["policy"]["revision"] != specs["policy"]["revision"]:
            raise ValueError("Hub lock checkpoint mismatch")
    else:
        lock = specs
        dump_json(lock_path, lock)
    snapshots = {}
    for key, spec in lock.items():
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Every Hub dependency must have an immutable revision")
        kwargs = {"allow_patterns": ["*.json", "*.txt", "*.model", "*.jinja"]} if key == "backbone" else {}
        snapshots[key] = Path(snapshot_download(**spec, **kwargs))
    return snapshots, lock


def configure_libero(assets):
    """Write an isolated local config before importing LIBERO (avoids input())."""
    import yaml
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero not installed")
    package_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    assets = assets / "assets" if (assets / "assets").is_dir() else assets
    config_dir = ROOT / ".libero"
    config_dir.mkdir(exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    paths = {"benchmark_root": str(package_root), "bddl_files": str(package_root / "bddl_files"),
             "init_states": str(package_root / "init_files"), "datasets": str(ROOT / ".cache/datasets"),
             "assets": str(assets)}
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    import libero.libero as core
    # HF LIBERO 0.1.4 uses this cache for object asset paths. Keep the pinned snapshot.
    core._assets_path_cache = str(assets)
    return core


def policy_observation(raw, language):
    import numpy as np
    from lerobot.envs.utils import preprocess_observation
    # Batch all state tensors before the official LIBERO quaternion processor.
    obs = {"pixels": {"image": raw["agentview_image"], "image2": raw["robot0_eye_in_hand_image"]},
           "robot_state": {"eef": {"pos": np.asarray(raw["robot0_eef_pos"])[None],
                                    "quat": np.asarray(raw["robot0_eef_quat"])[None]},
                           "gripper": {"qpos": np.asarray(raw["robot0_gripper_qpos"])[None]}}}
    batch = preprocess_observation(obs)
    batch["task"] = [language]
    return batch


def run_episode(env, init_state, policy, pre, post, env_pre, cfg, episode_id, seed, out, deadline):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.reset()
    started = time.perf_counter()
    env.seed(seed)
    env.reset()
    raw = env.set_init_state(init_state)
    for robot in env.env.robots:
        robot.controller.use_delta = True
    for _ in range(cfg["environment"]["settle_steps"]):
        raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    if env.check_success():
        raise RuntimeError("Initial state already satisfies task; invalid baseline episode")
    final = out / f"episode_{episode_id:03d}.mp4"
    partial = out / f"episode_{episode_id:03d}.partial.mp4"
    language = cfg["environment"]["expected_task_name"].replace("_", " ")
    language = cfg["environment"].get("task_language", language)
    success = False
    steps = 0
    inference_calls = 0
    inference_seconds = 0.0
    termination_reason = "horizon"
    def frame(obs):
        # Mirror exactly the official policy orientation, not just a vertical flip.
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    with imageio.get_writer(partial, fps=cfg["evaluation"]["video_fps"], codec="libx264",
                            macro_block_size=1, ffmpeg_log_level="error") as writer, \
            (out / f"episode_{episode_id:03d}.jsonl").open("x") as trace:
        writer.append_data(frame(raw))
        for steps in range(1, cfg["environment"]["max_steps"] + 1):
            if time.monotonic() > deadline:
                raise TimeoutError("Evaluation wall-time budget exceeded")
            observation = env_pre(policy_observation(raw, language))
            if tuple(observation["observation.state"].shape) != (1, 8):
                raise RuntimeError("Expected LIBERO state shape (1, 8)")
            with torch.inference_mode():
                torch.cuda.synchronize()
                inference_started = time.perf_counter()
                action = post(policy.select_action(pre(observation)))
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - inference_started
                # This protocol executes exactly one action per newly predicted chunk.
                inference_calls += 1
            action = action.detach().float().cpu().numpy()
            if action.shape != (1, 7) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid policy action shape/value: {action.shape}")
            raw, reward, done, _ = env.step(action[0].tolist())
            success = bool(env.check_success())
            writer.append_data(frame(raw))
            trace.write(json.dumps({"step": steps, "action": action[0].tolist(),
                                    "reward": float(reward), "done": bool(done),
                                    "success": success}) + "\n")
            if success or done:
                termination_reason = "success" if success else "environment_done"
                break
    # A decodable video, not merely a nonempty file, is required to complete an episode.
    with imageio.get_reader(partial) as reader:
        if reader.get_data(0).shape[:2] != (cfg["environment"]["image_size"],) * 2:
            raise RuntimeError("Video frame dimensions do not match the simulation")
        reader.get_data(steps)
    partial.rename(final)
    return {"status": "completed", "success": success, "steps": steps,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "policy_inference_calls": inference_calls,
            "policy_inference_seconds": round(inference_seconds, 6),
            "mean_inference_seconds": round(inference_seconds / inference_calls, 6),
            "termination_reason": termination_reason,
            "checkpoint_revision": cfg["policy"]["revision"],
            "config_sha256": config_fingerprint(cfg),
            "video_file": final.name, "video_sha256": sha256(final)}


def evaluate(cfg, out, smoke):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshots, lock = resolve_snapshots(cfg, out / "hub_revisions.json")
    configure_libero(snapshots["assets"])
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    e = cfg["environment"]
    suite = benchmark.get_benchmark_dict()[e["suite"]](task_order_index=e["task_order_index"])
    task = suite.get_task(e["task_id"])
    if task.name != e["expected_task_name"]:
        raise RuntimeError(f"Task ordering mismatch: {task.name}")
    if cfg["policy"]["n_action_steps"] != 1:
        raise ValueError("Stage 1 inference counting requires checkpoint-default n_action_steps=1")
    init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(e["task_id"])
    plan = episode_plan(cfg, smoke)
    if max(i for _, _, i in plan) >= len(init_states):
        raise ValueError("Insufficient initial states; refusing to repeat states")
    pcfg = PreTrainedConfig.from_pretrained(str(snapshots["policy"]))
    if not isinstance(pcfg, SmolVLAConfig):
        raise TypeError("Checkpoint did not resolve to a SmolVLA configuration")
    pcfg.device = "cuda"
    pcfg.use_amp = False
    pcfg.load_vlm_weights = False  # Full policy checkpoint already contains VLM weights.
    pcfg.vlm_model_name = str(snapshots["backbone"])
    pcfg.n_action_steps = cfg["policy"]["n_action_steps"]
    pcfg.num_steps = cfg["policy"]["num_steps"]
    # Strict loading prevents a partially random policy from being evaluated.
    policy = SmolVLAPolicy.from_pretrained(str(snapshots["policy"]), config=pcfg, strict=True)
    policy.to(device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=str(snapshots["policy"]),
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "tokenizer_processor": {"tokenizer_name": str(snapshots["backbone"])}})
    env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    dump_json(out / "manifest.json", {"config": cfg, "config_sha256": config_fingerprint(cfg),
              "hub_revisions": lock, "bddl_sha256": sha256(bddl), "init_states_sha256": sha256(init_file),
              "task_name": task.name, "task_language": task.language,
              "policy_frozen": all(not p.requires_grad for p in policy.parameters()),
              "policy_dtype": "float32", "command": sys.argv, "seed_state_plan": plan,
              "git": command_output(["git", "rev-parse", "HEAD"]).get("stdout"),
              "versions": dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions()
                                      if d.metadata["Name"]))})
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=e["image_size"],
                             camera_widths=e["image_size"],
                             camera_names=["agentview", "robot0_eye_in_hand"])
    episode_cfg = json.loads(json.dumps(cfg))
    episode_cfg["environment"]["task_language"] = task.language
    deadline = time.monotonic() + cfg["evaluation"]["max_wall_time_seconds"]
    rows = []
    try:
        for episode_id, seed, state_id in plan:
            logging.info("episode=%s seed=%s initial_state=%s", episode_id, seed, state_id)
            row = {"phase": "smoke" if smoke else "evaluation", "episode_id": episode_id,
                   "suite": e["suite"], "task_id": e["task_id"], "task_name": task.name,
                   "seed": seed, "initial_state_id": state_id}
            row.update(run_episode(env, init_states[state_id], policy, pre, post, env_pre,
                                   episode_cfg, episode_id, seed, out, deadline))
            row["config_sha256"] = config_fingerprint(cfg)
            append_result(out / "baseline.csv", row)
            rows.append(row)
            logging.info("completed episode=%s success=%s steps=%s seconds=%s", episode_id,
                         row["success"], row["steps"], row["runtime_seconds"])
    finally:
        env.close()
    if smoke:
        dump_json(out / "smoke.json", {**rows[0], "config_sha256": config_fingerprint(cfg),
                                       "hub_revisions": lock})
    else:
        dump_json(out / "summary.json", {"status": "completed", "episodes": len(rows),
                  "successes": sum(r["success"] for r in rows),
                  "failures": sum(not r["success"] for r in rows),
                  "infrastructure_errors": 0,
                  "mean_runtime_seconds": statistics.mean(r["runtime_seconds"] for r in rows),
                  "mean_episode_steps": statistics.mean(r["steps"] for r in rows),
                  "policy_inference_calls": sum(r["policy_inference_calls"] for r in rows),
                  "total_policy_inference_seconds": sum(r["policy_inference_seconds"] for r in rows),
                  "mean_inference_seconds": sum(r["policy_inference_seconds"] for r in rows) / sum(r["policy_inference_calls"] for r in rows),
                  "success_rate": sum(r["success"] for r in rows) / len(rows),
                  "scope": "one task; small pilot, not a suite-level result"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-record", type=Path, default=ROOT / "results/smoke/smoke.json")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.check_config:
        print(json.dumps({"status": "config_valid", "config_sha256": config_fingerprint(cfg),
                          "evaluation_plan": episode_plan(cfg), "smoke_plan": episode_plan(cfg, True)}, indent=2))
        return 0
    out = args.output_dir or ROOT / cfg["evaluation"]["output_dir"]
    if args.smoke:
        out = out / "smoke"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", cfg["environment"]["mujoco_gl"])
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "evaluation.log"), logging.StreamHandler()])
    report = preflight(cfg)
    dump_json(out / "preflight.json", report)
    logging.info("PREFLIGHT %s", json.dumps(report, sort_keys=True))
    if not report["passed"]:
        logging.error("BLOCKED: preflight failed. No episode executed and no success metric produced.")
        return 2
    if args.preflight:
        return 0
    try:
        csv_file = out / "baseline.csv"
        if csv_file.exists() and len(csv_file.read_text().splitlines()) > 1:
            raise ValueError("Existing episode results found; use a fresh output directory")
        if not args.smoke:
            marker = verify_smoke(args.smoke_record, cfg)
            lock = out / "hub_revisions.json"
            if lock.exists() and json.loads(lock.read_text()) != marker["hub_revisions"]:
                raise ValueError("Hub lock differs from smoke run")
            dump_json(lock, marker["hub_revisions"])
        evaluate(cfg, out, args.smoke)
    except Exception:
        logging.exception("BLOCKED: execution interrupted; completed CSV rows remain real results")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
