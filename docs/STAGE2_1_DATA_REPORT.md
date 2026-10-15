# Stage 2.1 data report

## Scope and evidence

Implemented on `stage2-memory`. Stage 1 files, experiments and model weights are
preserved. This stage contains no embeddings, retrieval, training or new robot
rollouts. Real native demonstration extraction ran locally on macOS; Google Cloud,
V100 preflight, GCS upload and simulator replay are **UNVERIFIED**.

Stage 1 evidence: `configs/baseline.yaml`, `README.md`, `SETUP.md`,
`results/manifest.json`, `results/summary.json`, `scripts/run_baseline.py`, and
`reproduction/README.md`. Exactly one task was used: suite `libero_object`,
zero-based task ID **0**, task order **0**, name
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`, instruction
“pick up the alphabet soup and place it in the basket”. The five successful
policy evaluations are distinct from the 50 expert demonstrations below.

## Source and measured native storage

The [official LIBERO downloader](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/utils/download_utils.py)
identifies `yifengzhu-hf/LIBERO-datasets` as its Hugging Face source. We download
only [this task's native HDF5](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5),
revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
This is not a LeRobot/OpenVLA regenerated dataset.

Measured source bytes: **780,145,352**. SHA256:
`42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d`.
Default extraction rejects any other checksum. `--allow-unverified-source` is an
explicit escape hatch for independently audited native-format variants and
synthetic tests, labels their provenance unknown, and still checks task/controller
metadata. Do not use it for pre-action or regenerated datasets.

HDF5 root `/data` contains **50** groups `demo_0` through `demo_49`.
Observed total **7,808** steps; lengths **136–196**, mean **156.16**.
All 50 pass schema, finite-value and duplicate-content checks. No rejections.

| Native dataset within each demo | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `obs/agentview_rgb` | T×128×128×3 | uint8 | main RGB, native OpenGL convention |
| `obs/eye_in_hand_rgb` | T×128×128×3 | uint8 | wrist RGB |
| `obs/ee_states` | T×6 | float64 | position XYZ + axis-angle XYZ |
| `obs/ee_pos`, `obs/ee_ori` | T×3 each | float64 | corresponding state components |
| `obs/gripper_states` | T×2 | float64 | two finger joint positions |
| `obs/joint_states` | T×7 | float64 | robot arm joint positions |
| `actions` | T×7 | float64 | relative OSC pose command + gripper |
| `robot_states` | T×9 | float64 | auxiliary robot vector; not the policy's 8D state |
| `states` | T×110 | float64 | flattened simulator state; not proprioception |
| `rewards`, `dones` | T | uint8 | terminal indicators; not verified success |

Metadata includes language, environment arguments, controller scaling, BDDL path,
`macros_image_convention=opengl`, `tag=libero-v1`, total and demo count. Episode
attributes include XML model, initial simulator state and sample count. Full
source attributes and an example schema are recorded in generated metadata.
The source file retains all auxiliary observations and episode attributes.

## Alignment and success interpretation

The [native writer](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)
steps the simulator before collecting images/proprioception. Its same-index
observation is therefore post-action. Extraction pairs `obs[i]` with
`actions[i+1]`, dropping the first action and last observation. There are T−1
usable pairs per trajectory; no boundary crossing. Indices refer to the stored
file, not original collection timestamps. No timestamps are invented. Controller
metadata records 20 Hz, but extraction does not claim a wall-clock timeline.

The writer assigns final rewards/dones to one without deriving them from a
success predicate. These are **not success labels**. We retain the official
expert-demonstration provenance assumption as
`assumed_expert_provenance_not_verified`, with verified success count **null**.
Alignment is justified by writer source and shape/index checks; physical replay
and successful completion of each demonstration remain **UNVERIFIED**. A new
success-filtered claim requires simulator replay and per-episode evidence.

## Stage 1 versus demonstrations

Stage 1 reads raw `agentview_image`, `robot0_eye_in_hand_image` at 256×256,
`robot0_eef_pos` (3), `robot0_eef_quat` (4, XYZW), and
`robot0_gripper_qpos` (2). The
[LeRobot v0.5.1 processor](https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/processor/env_processor.py)
converts quaternion to 3D axis-angle, concatenates 8D state as position,
orientation, finger positions, casts float32, and rotates both images 180°.
Policy keys are `observation.images.image`, `observation.images.image2`,
`observation.state`. Image preprocessing produces batched CHW floats.

Native extracted state has the same conceptual 8D ordering but stays float64;
RGB stays 128×128 HWC uint8 with no rotation. Upscaling cannot recreate Stage 1's
256×256 detail. Stage 2.2 must verify matching orientation and preprocessing on
actual rendered frames, including axis-angle branch behavior. No policy-ready
image equivalence is claimed here.

The [pinned checkpoint config](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
uses 7D actions, 8D state, MEAN_STD state/action normalization, visual IDENTITY,
and internal image padding/resizing to 512×512. Stage 1 uses its saved
pre/postprocessors, including action unnormalization. It generates 50-step
chunks, executes one action, discards 49 and replans. Internal padded model
vectors are not the environment's 7D commands.

Native actions are `[dx,dy,dz,dax,day,daz,gripper]`: scaled controller inputs,
not meters/radians directly and not absolute joint targets. Recorded
`OSC_POSE` config uses relative control, input range [-1,1], positional output
range ±0.05 m and rotational output range ±0.5 rad. The
[robosuite 1.4 OSC controller](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py)
uses world-frame end-effector position deltas and an axis-angle rotation update.
Its input clipping/scaling is separate from learned checkpoint normalization.
The [Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)
uses −1 open, +1 close. Stage 1 adds no gripper inversion or clipping.

Thus expert commands and postprocessed policy commands have compatible intended
controller semantics, but raw expert values must **not** be compared directly
to normalized model predictions. Preserve source values, then choose and verify
one comparison space in Stage 2.2. No new normalization is fit here.

## Intermediate format and split

Each partition is a compressed, chunked HDF5 file, supporting trajectory and
frame slicing without loading all images. Real outputs total about 416 MiB,
versus the 744 MiB native file. JSON is used only for metadata/manifests.

`/trajectories/demo_N` contains:

- `agentview_rgb`, `eye_in_hand_rgb`: (T−1,H,W,3) uint8, exact source pixels.
- `state`: (T−1,8), float64 concatenation of EE and gripper components.
- `joint_states`: (T−1,7), float64.
- `actions`: (T−1,7), float64, exact source `actions[1:]` values.
- `timestep`, `source_obs_index`: int64 0…T−2.
- `source_action_index`: int64 1…T−1.

Group attributes carry task ID, suite, task name, globally qualified trajectory
ID, source checksum and success status. Every row inherits its group's identity;
no repeated string arrays are needed. The retained source contains unshifted
simulator states and terminal flags; these are not misrepresented as aligned
policy inputs.

Seed **6758**, evaluation fraction **0.2**. Sort unique demo IDs by
SHA256(`seed:demo_id`); assign the first rounded fraction to evaluation. Result:
**40 train / 10 evaluation**, **6,193 / 1,565** aligned pairs, **7,758** total.
`split_manifest.json` records exact membership, source lengths, content hashes,
seed, algorithm and source checksum. Duplicate content is rejected before splitting.
This is held-out demonstration evaluation, not new robot success evaluation;
checkpoint training overlap is unknown, so it is not an unseen-policy-data claim.

Missing/malformed records fail closed. `--skip-invalid` explicitly quarantines
entire trajectories with reasons, recomputing the split over valid trajectories.
At least two valid trajectories are required. Outputs are staged, validated and
renamed into a fresh directory; existing outputs are never overwritten. Source
is opened read-only. Dataset and output hashes allow integrity checks.

## Validation and limitations

See `STAGE2_1_LOCAL_VALIDATION.json` for actual local results and action ranges.
Tests cover temporal shifts, exact values, source preservation, split determinism,
leakage, missing modalities, NaNs, bad shapes/types, duplicate content, wrong task,
wrong controller, unverified sources, output corruption, and notebook syntax/schema.
Stage 1's four tests also run unchanged. Cloud notebook cells have not executed
on the V100; syntax/schema checks are not cloud execution evidence.
