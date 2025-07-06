import torch
print(torch.__version__)                  # Should show 2.7.0+cu128
print(torch.cuda.is_available())          # Should be True
print(torch.cuda.get_device_name(0))      # Should say "NVIDIA GeForce RTX 4090"
