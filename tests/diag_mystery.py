
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from hcm import HCM, Config
from hcm.ops import unit
from hcm.data import overlapping_set

D = 128
X = overlapping_set(30, D, overlap=0.90, seed=11)
m = HCM(Config(d_x=D, a=0.02, capacity=500, beta=20.0, L=0, use_resid=False, seed=0))

print("=== 关键疑问：同一输入，写入码 与 查询码 是否相同？ ===")
# 先只 observe 前 5 条，逐步检查
codes_write = []
for i, x in enumerate(X):
    xe = m.enc.resid_write(x)          # 写入路径（会更新 mu）
    h_write = m.enc.encode(xe)
    codes_write.append(h_write.copy())
    m.observe(x)

# 现在查询：使用 resid_read（不更新 mu）
print("  i  |write-query cos| 写入了吗")
for i in range(6):
    xr = m.enc.resid_read(X[i])
    h_query = m.enc.encode(xr)
    c = float(h_query @ codes_write[i])
    wrote = m.log[i]["wrote"]
    print("  %2d |    %.4f     | %s" % (i, c, wrote))
print()
print("  use_resid=False 时 resid_write 返回 unit(x)，resid_read 返回 unit(x)")
print("  => 理论上 cos 应恒为 1.000。若不是，说明 encode 不确定（theta 在变？）")
print("  theta 非零元素数:", int((m.enc.theta != 0).sum()), " theta max:", float(np.abs(m.enc.theta).max()))
print()
print("=== 库里到底存了什么？ ===")
print("  写入条数:", m.E.n, "/30")
print("  前 6 条写入码的支撑集是否互不相同:")
for i in range(6):
    sup = tuple(np.where(m.E.M[:, i] > 0)[0])
    print("    i=%d 支撑集大小=%d  前5个索引=%s" % (i, len(sup), sup[:5]))
