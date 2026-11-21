import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 1) reproduce_all 加入 exp12
p = os.path.join(ROOT, "bench", "reproduce_all.py")
s = open(p, encoding="utf-8").read()
if "exp12_generator_sensitivity" not in s:
    s = s.replace("    ('双读出对照', 'bench\\\\exp11_readout.py'),",
                  "    ('双读出对照', 'bench\\\\exp11_readout.py'),\n"
                  "    ('生成器敏感性', 'bench\\\\exp12_generator_sensitivity.py'),")
    open(p, "w", encoding="utf-8").write(s)
    print("reproduce_all: exp12 added")

# 2) check_claims 校验 §4.11 的关键数字
c = os.path.join(ROOT, "tools", "check_claims.py")
cs = open(c, encoding="utf-8").read()
add = '''
    # 表 11：生成器敏感性（exp12）
    g12 = J("exp12_generator.json")
    def g12v(share, side, key):
        v = [r[side][key] for r in g12 if abs(r["share"] - share) < 1e-6]
        return v[0] if v else None
    expect("表11 share0.90 输入余弦", g12v(0.90, "resid_off", "code_cos"), 0.872, 0.002)
    expect("表11 share0.90 无残差召回", g12v(0.90, "resid_off", "recall"), 1, 0)
    expect("表11 share0.90 残差召回", g12v(0.90, "resid_on", "recall"), 25, 0)
    expect("表11 share0.90 无残差写入", g12v(0.90, "resid_off", "written"), 18, 0)
    expect("表11 share0.85 残差召回", g12v(0.85, "resid_on", "recall"), 30, 0)
    expect("表11 share0.95 残差召回", g12v(0.95, "resid_on", "recall"), 0, 0)
    expect("表11 share0.95 无残差写入", g12v(0.95, "resid_off", "written"), 6, 0)

'''
anchor = "    return problems, checked"
if "表11" not in cs:
    cs = cs.replace(anchor, add + anchor)
    open(c, "w", encoding="utf-8").write(cs)
    print("check_claims: table 11 added")
