# -*- coding: utf-8 -*-
"""exp29 v2 launcher: EXP29_OUT -> results/exp29_text_frontend_v2.json (D-053 six-session corpus).

Run on m2 via WMI/cmd so it survives ssh session exit (L-044):
  py -X utf8 -u bench/exp29_v2_m2.py --stage full --job-id full5m2v2
The env var is set HERE (before exp29_text_frontend is executed) because WMI-spawned
processes do not reliably inherit the caller's shell environment.
"""
import os
import runpy
import sys

os.environ["EXP29_OUT"] = os.path.join("results", "exp29_text_frontend_v2.json")
_run = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp29_text_frontend.py"), run_name="__main__")
