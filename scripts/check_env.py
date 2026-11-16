"""Environment sanity check. Every team member runs this after setup and pastes the
output into the Week 1 report so we know everyone's environment matches.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

REQUIRED = [
    "numpy", "pandas", "pyarrow", "sklearn", "xgboost",
    "torch", "transformers", "accelerate", "yaml", "dotenv", "kaggle",
]

ok = True


def report(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'OK ' if passed else 'FAIL'}] {label:<14} {detail}")


print(f"Python {platform.python_version()} on {platform.system()} ({sys.executable})")
report("python 3.11/12", sys.version_info[:2] in [(3, 11), (3, 12)], "3.11 or 3.12 required")

print("\nPackages:")
for name in REQUIRED:
    try:
        # kaggle authenticates on import and raises if kaggle.json is missing;
        # we only want to know it's installed here, credentials are checked below.
        if name == "kaggle":
            import importlib.util
            found = importlib.util.find_spec("kaggle") is not None
            report(name, found, "installed" if found else "pip install kaggle")
            continue
        mod = importlib.import_module(name)
        report(name, True, getattr(mod, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(name, False, f"{type(e).__name__}: {e}")

print("\nHardware:")
try:
    import torch

    if torch.cuda.is_available():
        report("cuda", True, f"{torch.cuda.get_device_name(0)} (torch CUDA {torch.version.cuda})")
    else:
        # Not a failure: SVM/TextCNN/XGBoost are fine on CPU; RoBERTa is just slower.
        print("  [WARN] cuda           not available -> CPU only (fine except RoBERTa = slow)")
except ImportError:
    pass

print("\nProject:")
try:
    from supplyiq import config

    report("supplyiq pkg", True, f"root={config.ROOT}")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    report("kaggle creds", kaggle_json.exists(), str(kaggle_json))
    missing = [f for f in config.M5_FILES if not (config.M5_RAW_DIR / f).exists()]
    if missing:
        print(f"  [TODO] M5 data       missing in {config.M5_RAW_DIR}: {', '.join(missing)}")
    else:
        print("  [OK ] M5 data        all files present")
except ImportError:
    report("supplyiq pkg", False, "run: pip install -e .")

print("\nRESULT:", "environment ready" if ok else "fix the FAIL lines above")
sys.exit(0 if ok else 1)
