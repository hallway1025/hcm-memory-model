# -*- coding: utf-8 -*-
"""exp36 smoke: tiny slice (overrides constants, uses temp output)."""
import os
base = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(base, 'exp36_b2_multitau.py')
src = open(target, encoding='utf-8').read()
src = src.replace('MIN_POS = 3000', 'MIN_POS = 20')
src = src.replace('N_TARGET = 200', 'N_TARGET = 8')
src = src.replace('NSEED = 3', 'NSEED = 1')
src = src.replace("exp36_b2_multitau.json", "exp36_smoke_tmp.json")
src = src.replace("MIN_COS = 0.5", "MIN_COS = 0.3")
g = {'__name__': '__main__', '__file__': target}
exec(compile(src, 'exp36_smoke', 'exec'), g)
