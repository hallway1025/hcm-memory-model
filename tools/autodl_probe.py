
import sys, os, subprocess
sys.path.insert(0, "src")
import numpy as np
from hcm.model import Config, HCM
print("import OK, np", np.__version__, "py", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch missing:", e)
r = subprocess.run([sys.executable, "tests/smoke.py"], capture_output=True, text=True)
print("smoke exit", r.returncode)
print(r.stdout[-1200:])
print(r.stderr[-400:] if r.returncode else "")
