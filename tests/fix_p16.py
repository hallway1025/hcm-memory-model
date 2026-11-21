import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'tools', 'make_manifest.py')
s = open(p, encoding='utf-8').read()

old_inc = "INCLUDE = ['README.md', 'docs', 'src', 'bench', 'tools', 'paper', 'results', 'refs/refs.jsonl', 'audit/AUDIT-REPORT.md']"
new_inc = ("# 覆盖全项目（第二轮审查 P1-6：早期清单只含 94 个文件，tests/ 未覆盖）。" + chr(10)
           + "INCLUDE = ['.']" + chr(10)
           + "SKIP_DIRS = {'__pycache__', 'cache', '.git', '_scratch'}" + chr(10)
           + "SKIP_FILES = {'SHA256SUMS.txt'}   # 清单不能包含自己")
if old_inc in s:
    s = s.replace(old_inc, new_inc)
    # 移除旧的重复 SKIP_DIRS 定义
    s = s.replace("SKIP_DIRS = {'__pycache__', 'cache', 'figures'}\n", '')
    s = s.replace("SKIP_DIRS = {'__pycache__', 'cache', 'figures'}", '')
    # 在 files() 里加 SKIP_FILES 过滤
    s = s.replace("                    if f.endswith('.pyc'): continue",
                  "                    if f.endswith('.pyc') or f in SKIP_FILES: continue")
    open(p, 'w', encoding='utf-8').write(s)
    print('manifest now covers whole project')
else:
    print('pattern not found, showing INCLUDE lines:')
    for ln in s.split(chr(10)):
        if 'INCLUDE' in ln or 'SKIP_DIRS' in ln: print('  ', ln)