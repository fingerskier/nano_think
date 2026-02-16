"""GPU/CUDA verification script for nano_think."""
import torch

print("=" * 60)
print("STEP 1: CUDA Verification")
print("=" * 60)
print(f"PyTorch version:       {torch.__version__}")
print(f"CUDA available:        {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE - stopping here")
    raise SystemExit(1)

print(f"CUDA version:          {torch.version.cuda}")
print(f"Device count:          {torch.cuda.device_count()}")
print(f"Current device:        {torch.cuda.current_device()}")
print(f"Device name:           {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"Total memory:          {props.total_memory / 1024**3:.1f} GB")
print(f"Compute capability:    {props.major}.{props.minor}")
free, total = torch.cuda.mem_get_info(0)
print(f"Free memory:           {free / 1024**3:.1f} GB")
print(f"cuDNN version:         {torch.backends.cudnn.version()}")
print(f"cuDNN enabled:         {torch.backends.cudnn.enabled}")

print()
print("=" * 60)
print("STEP 2: Tensor Operations on GPU")
print("=" * 60)
device = torch.device("cuda")
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)
c = a @ b
print(f"Matrix multiply (1024x1024): {c.shape} on {c.device}")
print(f"Result sample:  {c[0,:3].tolist()}")
cpu_t = torch.randn(256, 256)
gpu_t = cpu_t.to(device)
back = gpu_t.cpu()
print(f"CPU->GPU->CPU transfer: match={torch.allclose(cpu_t, back)}")
print("Tensor ops on GPU: PASS")

print()
print("=" * 60)
print("STEP 3: AMP (Mixed Precision) Test")
print("=" * 60)
with torch.amp.autocast("cuda", dtype=torch.float16):
    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)
    z = x @ y
    print(f"float16 autocast:  dtype={z.dtype}, device={z.device}")

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)
    z = x @ y
    print(f"bfloat16 autocast: dtype={z.dtype}, device={z.device}")
print("AMP mixed precision: PASS")

print()
print("=" * 60)
print("STEP 4: NanoThink Model on GPU")
print("=" * 60)
from nano_think.config import (
    ModelConfig, TransformerConfig, DiffuserConfig,
    StateSpaceConfig, MLAConfig, VectorStoreConfig,
)
from nano_think.model import NanoThink

model = NanoThink(
    model_cfg=ModelConfig(),
    transformer_cfg=TransformerConfig(),
    diffuser_cfg=DiffuserConfig(),
    ssm_cfg=StateSpaceConfig(),
    mla_cfg=MLAConfig(),
    vs_cfg=VectorStoreConfig(),
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters:  {n_params:,}")

model = model.to(device)
model.eval()
print(f"Model moved to:    {next(model.parameters()).device}")

input_ids = torch.randint(0, 32000, (1, 64), device=device)
with torch.no_grad():
    out = model(input_ids, use_vector_store=False)
logits = out["logits"]
router_w = out["router_weights"]
print(f"Input shape:       {input_ids.shape}")
print(f"Logits shape:      {logits.shape}  device={logits.device}")
print(f"Router weights:    {router_w.shape}  device={router_w.device}")
print(f"Router values:     {router_w[0,0].tolist()}")
print("NanoThink forward pass on GPU: PASS")

# Cleanup
del model, out, logits, router_w, input_ids
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info(0)
print(f"GPU memory after cleanup: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB free")

print()
print("=" * 60)
print("ALL GPU/CUDA CHECKS PASSED")
print("=" * 60)
