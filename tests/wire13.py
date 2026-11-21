import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'bench', 'reproduce_all.py')
s = open(p, encoding='utf-8').read()
adds = [("    ('生成器敏感性', 'bench\\\\exp12_generator_sensitivity.py'),",
         "    ('生成器敏感性', 'bench\\\\exp12_generator_sensitivity.py'),\n    ('置信度信号', 'bench\\\\exp13_confidence.py'),")]
if 'exp13_confidence' not in s:
    s = s.replace("    ('生成器敏感性', 'bench\\\\exp12_generator_sensitivity.py'),",
                  "    ('生成器敏感性', 'bench\\\\exp12_generator_sensitivity.py'),\n"
                  "    ('置信度信号', 'bench\\\\exp13_confidence.py'),")
    open(p, 'w', encoding='utf-8').write(s)
    print('reproduce_all: exp13 added')
else:
    print('exp13 already in reproduce_all')

c = os.path.join(ROOT, 'tools', 'check_claims.py')
cs = open(c, encoding='utf-8').read()
block = '''
    # 表 12：置信度信号（exp13）
    cf = J("exp13_confidence.json")
    expect("表12 corr(m1,correct)", cf["corr_m1"], -0.0697, 0.005)
    expect("表12 AUC(m1)", cf["auc_m1"], 0.458, 0.005)
    expect("表12 AUC(raw)", cf["auc_raw"], 1.000, 0.005)
'''
if '表12' not in cs:
    cs = cs.replace('    return problems, checked', block + '    return problems, checked')
    open(c, 'w', encoding='utf-8').write(cs)
    print('checker: table 12 added')