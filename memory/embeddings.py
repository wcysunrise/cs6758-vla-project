"""Frozen ResNet-18 and deterministic feature construction, independent of SmolVLA."""
import os
from pathlib import Path
import numpy as np


def unit_rows(x, allow_zero=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite feature matrix')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not allow_zero and (norms < 1e-12).any():
        raise ValueError('Zero feature vector has undefined cosine')
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def fit_state_stats(states):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 8 or not len(states) or not np.isfinite(states).all():
        raise ValueError('Expected nonempty finite (N,8) training states')
    mean, std = states.mean(0), states.std(0)
    return mean, np.where(std < 1e-6, 1.0, std)


def standardize_state(states, mean, scale):
    x, mean, scale = [np.asarray(a, dtype=np.float64) for a in (states, mean, scale)]
    if x.ndim != 2 or x.shape[1] != 8 or mean.shape != (8,) or scale.shape != (8,):
        raise ValueError('Invalid state/statistics shape')
    if not all(np.isfinite(a).all() for a in (x, mean, scale)) or (scale <= 0).any():
        raise ValueError('Invalid state/statistics values')
    return (x - mean) / scale


def combine_features(visual, states, mean, scale, mode='visual_state', visual_weight=1., state_weight=1.):
    if mode not in ('visual', 'visual_state'):
        raise ValueError('Unknown retrieval mode')
    weights = np.asarray([visual_weight, state_weight], dtype=float)
    if not np.isfinite(weights).all() or visual_weight <= 0 or state_weight < 0:
        raise ValueError('visual_weight must be positive; state_weight nonnegative')
    v = unit_rows(visual)
    if mode == 'visual':
        return v
    z = unit_rows(standardize_state(states, mean, scale), allow_zero=True)
    if len(v) != len(z):
        raise ValueError('Feature/state batch mismatch')
    return unit_rows(np.concatenate((np.sqrt(visual_weight) * v, np.sqrt(state_weight) * z), axis=1))


class VisualEncoder:
    """IMAGENET1K_V1, average pool before classifier, 512D; never trainable."""
    def __init__(self, device='cpu', threads=4):
        os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parents[1] / '.cache/torch'))
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights
        from .dataset_utils import sha256
        if torch.__version__.split('+')[0] != '2.7.1' or torchvision.__version__.split('+')[0] != '0.22.1':
            raise RuntimeError('Use pinned torch 2.7.1 / torchvision 0.22.1 for preprocessing reproducibility')
        if device not in ('cpu', 'cuda'):
            raise ValueError('Supported devices: cpu, cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(threads)
        torch.manual_seed(6758)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = device
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().requires_grad_(False).to(device)
        self.transform = weights.transforms(antialias=True)
        checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
        self.metadata = {'architecture': 'resnet18_avgpool', 'dimension': 512,
                         'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'weights_url': weights.url,
                         'weights_sha256': sha256(checkpoint), 'torch': torch.__version__,
                         'torchvision': torchvision.__version__, 'device': device,
                         'preprocessing': 'native_opengl HWC uint8 RGB -> rotate180 -> tensor CHW -> resize256 bilinear antialias -> center_crop224 -> /255 -> ImageNet mean/std',
                         'imagenet_mean': [0.485, 0.456, 0.406], 'imagenet_std': [0.229, 0.224, 0.225],
                         'frozen': True, 'dtype': 'float32', 'seed': 6758}

    def encode(self, images):
        import torch
        x = np.asarray(images)
        if x.ndim != 4 or x.shape[-1] != 3 or min(x.shape[1:3]) < 1 or x.dtype != np.uint8 or len(x) == 0:
            raise ValueError('Expected nonempty native OpenGL (N,H,W,3) uint8 RGB')
        # Same transform for native dataset and raw LIBERO simulator frames.
        x = torch.from_numpy(x[:, ::-1, ::-1, :].copy()).permute(0, 3, 1, 2)
        x = self.transform(x).to(self.device)
        with torch.inference_mode():
            result = self.model(x).cpu().numpy()
        if result.shape != (len(images), 512):
            raise RuntimeError('Unexpected encoder shape')
        return unit_rows(result)


def observation_state(observation):
    """Raw LIBERO XYZW quaternion to the Stage 1 axis-angle convention."""
    p = np.asarray(observation['robot0_eef_pos'], dtype=np.float64)
    q = np.asarray(observation['robot0_eef_quat'], dtype=np.float64)
    g = np.asarray(observation['robot0_gripper_qpos'], dtype=np.float64)
    if p.shape != (3,) or q.shape != (4,) or g.shape != (2,) or not np.isfinite(np.r_[p,q,g]).all():
        raise ValueError('Malformed raw LIBERO state')
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-4):
        raise ValueError('Quaternion must be unit length')
    w = np.clip(q[3], -1., 1.)
    den = np.sqrt(max(0., 1. - w*w))
    aa = q[:3] * (2. * np.arccos(w) / den) if den > 1e-10 else np.zeros(3)
    return np.r_[p, aa, g]
