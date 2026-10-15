# CS 6758: Frozen SmolVLA + LIBERO baseline

Stage 1 establishes a measured simulation reference for later visual robustness
experiments. The LIBERO-specific pretrained policy is frozen; no fine-tuning or
new learning algorithm is used.

## Measured result

**5/5 successes (100%)** on `libero_object`, zero-based task ID 0,
task order 0: **pick up the alphabet soup and place it in the basket**.
One independent smoke episode also succeeded and is excluded from this rate.
All five formal episodes terminated on the actual LIBERO success predicate.

| Episode | Seed | Fixed state | Outcome | Steps | Wall time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1000 | 0 | Success | 165 | 190.383 |
| 2 | 1001 | 1 | Success | 149 | 166.096 |
| 3 | 1002 | 2 | Success | 198 | 212.094 |
| 4 | 1003 | 3 | Success | 160 | 171.396 |
| 5 | 1004 | 4 | Success | 153 | 164.519 |

- Mean episode wall time: 180.898 s.
- Mean length: 165.0 control steps.
- Inference calls: 825; total synchronized inference time: 701.055 s.
- Weighted mean inference: 0.8498 s/call.
- Formal run: 5 completed, 0 task failures, no infrastructure errors within these five episodes.
  Earlier startup errors are preserved separately and are excluded from robot outcomes.

## System and protocol

The existing single V100-SXM2-16GB runtime ran Python 3.12.13. A project venv
contains torch 2.7.1+cu126, torchvision 0.22.1+cu126, LeRobot 0.5.1, hf-libero
0.1.4 and MuJoCo 3.3.2. The working system torch installation was preserved.
CUDA matrix multiplication, pip check, EGL rendering and strict checkpoint
loading passed. Full resolved versions are in the result manifests.

Checkpoint: `HuggingFaceVLA/smolvla_libero` at
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
Main/wrist 256x256 RGB and 8D robot state use official LeRobot processing and
checkpoint normalization. Output is a 7D relative OSC action. FP32, AMP off,
10 flow steps, chunk length 50, execute first action and discard 49, query again
every environment step. Horizon 280 steps plus 10 no-op settling steps.
Smoke: seed 1005/state 5. Formal: seeds 1000-1004/states 0-4. No parallel workers.
Simulated control/video is 20 Hz; measured inference is slower than real time.

## Reproduce

Follow [SETUP.md](SETUP.md) for the isolated Linux CUDA environment. From the
project root, activate `.venv`, then use a **fresh output directory**:

```bash
python scripts/run_baseline.py --check-config
python -m unittest discover -s tests -v
RUN_DIR="results/runs/reproduction_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_baseline.py --preflight --output-dir "$RUN_DIR"
python -u scripts/run_baseline.py --smoke --output-dir "$RUN_DIR"
python -u scripts/run_baseline.py --output-dir "$RUN_DIR" --smoke-record "$RUN_DIR/smoke/smoke.json"
```

The alternative [Colab Enterprise Notebook](notebooks/stage1_baseline.ipynb)
calls the same scripts, creates a fresh directory and exports all new artifacts.
Existing episode rows cannot be overwritten. Export and verify results before
stopping the runtime. No new infrastructure is created by the code.

## Actual artifacts

- [Per-episode CSV](results/baseline.csv) and [summary](results/summary.json).
- Formal videos: [1](results/episode_001.mp4), [2](results/episode_002.mp4), [3](results/episode_003.mp4), [4](results/episode_004.mp4), [5](results/episode_005.mp4).
- [Raw formal console log](results/evaluation_console.log), [structured evaluation log](results/evaluation.log).
- [Environment preflight](results/preflight.json), [run manifest](results/manifest.json),
  [Hub revisions](results/hub_revisions.json), [resolved packages](results/installed_versions.txt).
- [Independent smoke record](results/smoke/smoke.json) and its CSV, MP4 and trace.
- [Artifact verification](results/verification/artifact_checks.json),
  [offline test log](results/verification/offline_checks.log), and
  [executed source provenance](reproduction/README.md).

Each MP4 is real simulator footage and decodes fully. CSV checksums match video
bytes; traces have exactly the recorded number of actions and the final success
signal. Videos contain an initial frame plus one frame per policy step.
Dependencies, weights, caches, credentials and virtual environments are excluded
from the source/results package.

## Limitations and next step

Five episodes meet the minimum requested pilot target. Ten would extend GPU
runtime substantially; the fixed five-episode protocol was retained. This is one
task and five fixed states, not a full benchmark or a robustness evaluation.
The exact two-sided 95% binomial interval for 5/5 is approximately 47.8%-100%,
and these states are not a representative random benchmark sample. Seeds do not
guarantee bitwise determinism across hardware or simulator versions.

The pipeline is operational for paired Stage 2 pilot comparisons. Start with
one controlled brightness or occlusion perturbation using identical seeds and
states. The observed ceiling also motivates a harder task and a larger sample
before general claims. Preserve this baseline unchanged.

## Delivery status

Source and genuine results are prepared for
[wcysunrise/cs6758-vla-project](https://github.com/wcysunrise/cs6758-vla-project).
The user explicitly authorized public upload on October 14, 2026 (America/New_York).
This repository publishes the reviewed code, configuration, reproducible Notebook,
genuine measured results, all five formal videos and the independent smoke
artifacts. GitHub delivery uses the existing account's browser interface. The
existing GPU runtime was verified Stopped after export. No cloud credentials,
model weights, virtual environments or caches are included.
