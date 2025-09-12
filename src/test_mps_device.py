"""Test script to verify MPS device detection."""
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Local imports after path setup
from src.segmentation_utils.device_utils import (  # noqa: E402
    get_best_device,
    print_device_info,
    supports_mixed_precision,
    is_mps_device
)


def main():
    """Test MPS device detection functionality."""
    print("\n" + "="*60)
    print("MPS DEVICE DETECTION TEST")
    print("="*60 + "\n")
    
    # Get device info
    selected_device = get_best_device()
    
    # Print device selection
    print(f"Selected device: {selected_device}")
    print(f"Is MPS device: {is_mps_device(selected_device)}")
    print(f"Supports mixed precision: "
          f"{supports_mixed_precision(selected_device)}")
    
    # Print detailed info
    print_device_info()
    
    # Test device allocation
    print("\nTesting device allocation...")
    try:
        import torch
        device = torch.device(selected_device)
        
        # Create a small tensor on the device
        x = torch.randn(2, 3, device=device)
        print(f"✓ Successfully created tensor on {device}")
        print(f"  Tensor shape: {x.shape}")
        print(f"  Tensor device: {x.device}")
        
        # Test computation
        y = x @ x.T
        print(f"✓ Successfully performed computation on {device}")
        print(f"  Result shape: {y.shape}")
        
    except Exception as e:
        print(f"✗ Error testing device: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()