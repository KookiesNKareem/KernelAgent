"""Conservative cache identity. Different hardware/software requires a new check."""
import importlib.metadata
import platform
import torch

CACHE_VERSION = 2


def environment():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu.append({'name': p.name, 'capability': [p.major, p.minor], 'memory': p.total_memory})
    driver = None
    if gpu:
        import subprocess
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                   capture_output=True, text=True, check=True, timeout=3)
            driver = sorted(set(probe.stdout.strip().splitlines()))
        except (OSError, subprocess.SubprocessError):
            pass
    return {'driver': driver, 'cache_version': CACHE_VERSION, 'python': platform.python_version(),
            'torch': str(torch.__version__), 'triton': version('triton'), 'cuda': torch.version.cuda,
            'platform': platform.system(), 'gpu': gpu}


def signature(args):
    return [{'shape': list(a.shape), 'dtype': str(a.dtype), 'device': str(a.device),
             'stride': list(a.stride()), 'offset': a.storage_offset(), 'requires_grad': a.requires_grad}
            if isinstance(a, torch.Tensor) else {'scalar': a} for a in args]
