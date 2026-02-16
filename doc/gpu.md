# GPU Support

## Install Steps

 1. Reinstall PyTorch with CUDA support

 Replace the CPU-only PyTorch with the CUDA 12.8 build (compatible with your CUDA 13.0 driver):

 pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

 2. (Optional) Pin CUDA index in pyproject.toml

 Add a comment or [tool.pip] note so future pip install -e . doesn't accidentally pull the CPU build again. Alternatively, add a
 requirements-gpu.txt:

 --index-url https://download.pytorch.org/whl/cu128
 torch>=2.2

 File: C:\dev\fingerskier\agent\nano_think\pyproject.toml — no code change needed; the torch>=2.2 dependency is fine as-is since PyTorch version   
 selection depends on the pip index URL, not the version specifier.

 3. Verify GPU is detected

 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

 Should print True and your RTX GPU name.

 4. Run pretrain_experts to confirm

 python scripts/pretrain_experts.py --epochs 1

 Should print Using device: cuda instead of Using device: cpu.