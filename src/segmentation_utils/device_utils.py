"""Device utilities for handling CUDA, MPS, and CPU devices."""
import torch


def get_best_device() -> str:
    """
    Get the best available device in order of preference: CUDA > MPS > CPU.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "mps_available": (
            torch.backends.mps.is_available()
            if hasattr(torch.backends, "mps")
            else False
        ),
        "mps_built": (
            torch.backends.mps.is_built()
            if hasattr(torch.backends, "mps")
            else False
        ),
        "selected_device": get_best_device()
    }
    
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["cuda_memory_gb"] = props.total_memory / 1e9
    
    return info


def is_mps_device(device: str) -> bool:
    """Check if the device is MPS."""
    return device == "mps" or (
        isinstance(device, torch.device) and device.type == "mps"
    )


def supports_mixed_precision(device: str) -> bool:
    """
    Check if the device supports mixed precision training.
    
    Note: MPS has limited support for mixed precision as of PyTorch 2.0+
    """
    if device == "cuda":
        return True
    elif device == "mps":
        # MPS support for autocast is limited
        # Check PyTorch version for better MPS support
        return (
            hasattr(torch.cuda.amp, "autocast") and
            torch.__version__ >= "2.0.0"
        )
    else:
        return False


def get_autocast_context(device: str, enabled: bool = True):
    """
    Get the appropriate autocast context for the device.
    
    Args:
        device: Device string
        enabled: Whether to enable autocast
        
    Returns:
        Autocast context manager
    """
    if device == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    elif device == "mps" and supports_mixed_precision(device):
        # For MPS, we use cpu autocast which works with MPS tensors
        return torch.cpu.amp.autocast(enabled=enabled)
    else:
        # No-op context for CPU or when mixed precision is not supported
        class NoOpContext:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
        return NoOpContext()


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    print("\n" + "="*50)
    print("DEVICE INFORMATION")
    print("="*50)
    print(f"Selected device: {info['selected_device'].upper()}")
    
    if info['cuda_available']:
        print("CUDA available: Yes")
        print(f"CUDA devices: {info['cuda_device_count']}")
        print(f"GPU: {info['cuda_device_name']}")
        print(f"GPU Memory: {info['cuda_memory_gb']:.2f} GB")
    else:
        print("CUDA available: No")
    
    if info['mps_available']:
        print("MPS available: Yes")
        print(f"MPS built: {'Yes' if info['mps_built'] else 'No'}")
    else:
        print("MPS available: No")
    
    print("="*50 + "\n")