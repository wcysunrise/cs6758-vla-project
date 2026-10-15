# Stage 2.2 memory construction and retrieval

## Implementation

This stage adds a frozen visual encoder, an exact-search memory database and two
retrieval modes. It neither loads SmolVLA nor changes its weights, performs
training, or intervenes in policy actions. Stage 1 and Stage 2.1 source/data are
preserved. The existing Stage 2.1 artifacts are used without extraction.

The task remains `libero_object`, task ID 0, order 0:
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`.
Input is `data/stage2_1_verified/`. Build validates source provenance, file hashes,
all arrays, exact original deterministic split membership and 40/10 counts before
encoding. Only training HDF5 groups supply memory rows or state statistics.
Reading held-out data for integrity checking does not fit any statistics.

## Encoder and features

The visual encoder is torchvision **0.22.1 ResNet-18**, with explicit
`ResNet18_Weights.IMAGENET1K_V1`, PyTorch **2.7.1**. The classification layer is
replaced by identity; average-pooled output is **512D**. Parameters are frozen,
model is in evaluation mode, and inference uses FP32 `inference_mode`. No random
encoder fallback is permitted. Weight SHA256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Architecture, library/build versions, weight URL/hash and preprocessing are saved
in `memory/metadata.json`; query encoders must match the recorded configuration.

The default camera is `agentview_rgb`; `--camera eye_in_hand_rgb` creates a
separate wrist-camera memory. This implementation uses one camera per memory,
not a two-camera fusion. Both dataset and raw simulator inputs must be native
OpenGL HWC uint8 RGB. Rotate 180°, convert CHW, resize short edge to 256 with
bilinear antialiasing, center crop 224, rescale to [0,1], and apply ImageNet
mean/std. The rotation follows the existing Stage 1 convention; the subsequent
[ResNet weight transforms](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)
are shared between build and query. Do not supply already rotated, normalized
or policy-preprocessed images. Native 128×128 training images and live 256×256
images follow the same transform, but matching pixel detail is not guaranteed.

Visual-only: `L2(ResNet(image))`, 512D, cosine search.

Visual + state: fit an 8D population mean and standard deviation using all **6,193
training frames only** (frame-weighted, not trajectory-weighted). Constant
components with std < 1e-6 use scale 1. No clipping or test-data refit. Let
`v=L2(visual)` and `s=L2((state-mean)/scale)`. Combined vector is
`L2(concat(sqrt(w_visual)*v, sqrt(w_state)*s))`, **520D**. Defaults are 1/1;
visual weight must be positive and state weight nonnegative. A zero standardized
state stays zero. For nonzero blocks the final dot product is a weighted average
of block cosine similarities; a zero state has no directional contribution.
Changing weights affects search only, so no re-encoding is necessary.

Raw state adapter accepts `robot0_eef_pos` (3), unit XYZW `robot0_eef_quat` (4),
and `robot0_gripper_qpos` (2), using the Stage 1 quaternion-to-axis-angle
convention. Native extracted states already have the matching 8D ordering.
Axis-angle discontinuities and unbalanced positional/orientation relevance are
limitations of this deliberately simple baseline, not learned invariances.

## Database and search

`memory/demo_memory.npz` contains 6,193 rows from exactly 40 training trajectories.
The NPZ uses compressed numeric/string arrays with `allow_pickle=False` loading:

| Array | Shape / dtype |
| --- | --- |
| `visual_features` | N×512, float32, unit norm |
| `states` | N×8, original float64 |
| `actions` | N×7, original float64 |
| `state_mean`, `state_scale` | 8 each, float64 |
| `demo_ids`, `trajectory_ids` | N Unicode strings |
| `timesteps`, `task_ids` | N int64 |
| `source_obs_indices`, `source_action_indices` | N int64 |

Task name/suite/order live in metadata; image references are reconstructed as
`train_demonstrations.hdf5:/trajectories/<demo_id>/<camera>[timestep]`.
This avoids duplicating images. Keep the Stage 2.1 HDF5 files with the memory.
`dataset_directory_hint` is relative to the build location; moved exports use
`export_locations.json` and their included dataset directory instead.

No further action shift is performed. Memory row t preserves Stage 2.1's exact
`actions[t]`, already paired with original source observation t and action t+1.
Actions remain unnormalized relative OSC commands; feature normalization never
changes their values. Stage 2.1 provenance-based success assumptions remain in
metadata and are not promoted to simulator-verified successes.

Retrieval normalizes the query using saved training statistics, computes all
memory dot products, and sorts descending with stable memory-row tie breaking.
Top-1 is the first element of Top-5. Neighbors need not be different trajectories;
consecutive frames may dominate. Results include cosine, exact action values,
trajectory/demo IDs, timestep, source action index, task metadata and image
reference. The low-level `search_features` API requires features from the recorded
encoder; the raw-image API enforces the encoder match automatically.

Loading checks SHA256, schema/shapes, finite values, unit visual norms, original
train membership, unique row identities, ordering, source action alignment and
recomputes training-only state statistics. Fresh output paths prevent overwriting
memory artifacts. The full NPZ and full held-out JSON are Git-ignored; compact
metadata and actual local validation evidence are committed.

## Validation and interpretation

Local CPU validation completed: **28 tests passed, zero skips**. Full memory
build took **155.09 s**, and both-mode held-out encoding/search took **35.81 s**
on this Mac (not GPU timings). All **15,650** returned neighbor identities and
actions were checked against the memory. All prior-stage tracked files and
Stage 2.1 HDF5 hashes were unchanged.

| Diagnostic (all 1,565 held-out frames) | Visual-only | Visual + state |
| --- | --- | --- |
| Mean Top-1 cosine | 0.988008 | 0.964852 |
| Mean absolute Top-1 action difference | 0.101093 | 0.096065 |
| Held-out neighbors leaked into memory | 0 | 0 |

Cosines from different feature spaces are not directly comparable. The action
diagnostic averages all seven controller-input components, including gripper;
it is not a physical distance and was not used to tune the default weights.

Actual CPU results are recorded in `STAGE2_2_LOCAL_VALIDATION.json` and the small
`STAGE2_2_RETRIEVAL_EXAMPLES.json`. The full diagnostics file is
`memory/heldout_retrieval.json`. It queries every one of **1,565 held-out frames**
from the 10 evaluation trajectories in both modes. Zero returned neighbors may
belong to an evaluation trajectory. Tests include encoder dimensions/frozen state,
normalization, deterministic features, serialization, checksum errors, cosine
ranking, Top-1/Top-5, weighting, invalid queries, action alignment and leakage.

Mean cosine and mean absolute action distance are descriptive checks, not
retrieval accuracy or success-rate claims. Every episode belongs to the same task;
there is no unique ground-truth nearest neighbor. Similar frames can have different
valid actions. ImageNet features are not robotics-specialized. Demonstrations may
have been in SmolVLA pretraining, so the held-out split is only held out from this
memory database. CPU/CUDA floating-point differences can affect nearly tied ranks;
no cross-device bitwise equivalence is claimed.

V100/CUDA inference, fresh Colab execution, GCS transfer, live simulator adapter
validation and demonstration success replay remain **UNVERIFIED** locally.
See the handoff for cloud execution and Stage 2.3 prerequisites.
