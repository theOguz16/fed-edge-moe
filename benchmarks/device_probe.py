import os
import platform
import torch

print("=== SYSTEM ===")
print("OS:", platform.platform())
print("Machine:", platform.machine())
print("Logical CPUs:", os.cpu_count())

print("\n=== PYTORCH ===")
print("PyTorch:", torch.__version__)
print("CPU threads:", torch.get_num_threads())
print("Interop threads:", torch.get_num_interop_threads())

print("\n=== ACCELERATORS ===")
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print(
        "CUDA memory GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )

if torch.backends.mps.is_available():
    print("Apple MPS available")
