#!/usr/bin/env python3
"""检查环境是否配置正确"""
import importlib
import os
import sys

required_packages = [
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("datasets", "datasets"),
]

print("=" * 60)
print("Auto SLR — 环境检查")
print("=" * 60)

all_ok = True
for module_name, pkg_name in required_packages:
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "ok")
        print(f"  ✅ {pkg_name:25s} {ver}")
    except ImportError:
        print(f"  ❌ {pkg_name:25s} 未安装")
        all_ok = False

print()
print("--- CUDA 检查 ---")
if torch.cuda.is_available():
    print(f"  ✅ CUDA 可用")
    print(f"     GPU: {torch.cuda.get_device_name(0)}")
    print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"  ⚠️  CUDA 不可用，将使用 CPU（训练会极慢）")

print()
print("--- 环境变量检查 ---")
env_vars = ["CLEF_TAR_BASE", "BASE_MODEL_PATH", "OPENAI_API_KEY"]
for var in env_vars:
    val = os.environ.get(var, "") or os.getenv(var, "")
    if val:
        print(f"  ✅ {var:20s} = {val}")
    else:
        print(f"  ⚠️  {var:20s} 未设置（可从 .env 加载）")

print()
if all_ok:
    print("✅ 所有核心依赖已就绪")
else:
    print("⚠️  部分依赖缺失，请运行: pip install -r requirements.txt")

print("=" * 60)
