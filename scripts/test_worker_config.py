import os
import socket
import subprocess

import ray
import torch

print("=" * 80)
print("HOST INFO")
print("=" * 80)

print("hostname:", socket.gethostname())
print("pid:", os.getpid())

print("\n" + "=" * 80)
print("RAY INFO")
print("=" * 80)

ray.init(address="auto")

print("cluster resources:")
print(ray.cluster_resources())

print("\navailable resources:")
print(ray.available_resources())

print("\n" + "=" * 80)
print("PYTORCH CUDA INFO")
print("=" * 80)

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print("name:", torch.cuda.get_device_name(i))
        print("capability:", torch.cuda.get_device_capability(i))

        props = torch.cuda.get_device_properties(i)

        print("total memory GB:", round(props.total_memory / 1024**3, 2))
        print("multi processor count:", props.multi_processor_count)

print("\n" + "=" * 80)
print("NCCL INFO")
print("=" * 80)

try:
    print("nccl version:", torch.cuda.nccl.version())
except Exception as e:
    print("nccl unavailable:", e)

print("\n" + "=" * 80)
print("NVIDIA-SMI / TEGRA")
print("=" * 80)

commands = [
    ["nvidia-smi"],
    ["tegrastats", "--interval", "1000"],
]

for cmd in commands:
    try:
        print(f"\nRunning: {' '.join(cmd)}")
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=3)
        print(out.decode())
    except Exception as e:
        print("failed:", e)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)