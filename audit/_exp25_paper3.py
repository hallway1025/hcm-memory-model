# -*- coding: utf-8 -*-
import io
NL = '\r\n'
def block(s):
    return s.replace('\n', NL)
P = 'paper/PAPER.md'
t = io.open(P, encoding='utf-8').read()
o = '| iid 高斯键（无共享结构） | 2, 2, 2, 2（无坍缩） | 98%, 100%, 100%, 100% |'
n = '| iid 高斯键（无共享结构） | 2, 2, 2, 2（无坍缩） | 98.5%, 100%, 100%, 100% |'
assert o in t; t = t.replace(o, block(n), 1)
o = '| 二值 k-稀疏码（超几何原子） | 2, 2, 2, 2（近免疫） | 89%, 100%, 88%, 88% |'
n = '| 二值 k-稀疏码（超几何原子） | 2, 2, 2, 2（近免疫） | 89.5%, 100%, 88%, 88% |'
assert o in t; t = t.replace(o, block(n), 1)
o = '| 16 簇共享质心键（真实记忆共享要旨的还原） | **2, 5, 5, 5** | **73%, 8%, 0%, 4%** |'
n = '| 16 簇共享质心键（真实记忆共享要旨的还原） | **2, 5, 5, 5** | **73.5%, 8.5%, 0%, 4%** |'
assert o in t; t = t.replace(o, block(n), 1)
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('percentages fixed')
