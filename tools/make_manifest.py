'''生成交付物 SHA256 清单（替代缺失的 git 提交哈希）。
对抗评审与洁净复现者都指出：项目不是 git 仓库，论文里「以提交哈希为准」无法兑现。
本脚本为所有交付文件生成 SHA256，作为可引用的冻结凭证。

2026-11-21 修：此前 INCLUDE=['.'] 从项目根递归，把 .venv-repro 的 466 个第三方
pip 包与 5 条 Windows 系统缓存扫进清单（1363 条里仅 892 条是交付物）。
现改为**白名单目录**，只收项目自身的交付物。
'''
import hashlib, os, subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 白名单：只收这些顶层目录（项目自身交付物）
INCLUDE_DIRS = ['src', 'bench', 'results', 'tools', 'refs', 'tests',
                'docs', 'repro', 'paper', 'audit', 'notes']
# 根目录下的单文件（交付相关）
INCLUDE_ROOT_FILES = ['README.md']

SKIP_DIRS = {'__pycache__', 'cache', '.git', '_scratch',
             '.venv', '.venv-repro', 'node_modules', '.pytest_cache', '.mypy_cache',
             'site-packages', 'dist', 'build', '.idea', '.vscode'}
# 不随仓库分发的数据目录：真实聊天语料含姓名/手机号，公开仓库不含，清单也不该列
SKIP_DIR_PATHS = {'bench/corpus', 'bench/corpus_oldcorpus', 'bench/corpus_public'}
PIP_TEMP_PREFIX = 'pip-unpack-'
SKIP_FILES = {'SHA256SUMS.txt', 'citations-report.json', 'citations-report.md',
              '.gitignore', '.DS_Store', 'Thumbs.db'}
# 复现会重写的派生结果（每次跑值都变），不入冻结清单
SKIP_RESULT_FILES = {'ablation.json', 'ablation2.json', 'durability.json',
                     'exp2c_capacity_edge.json'}
# 复现会重新导出的文档派生物（含时间戳/生成器元数据，每次重跑都变）
SKIP_PAPER_FILES = {'PAPER_final.md', 'PAPER_EN_final.md',
                    'HCM_PAPER.docx', 'HCM_PAPER.html', 'HCM_PAPER.pdf',
                    'HCM_PAPER_EN.docx', 'HCM_PAPER_EN.html', 'HCM_PAPER_EN.pdf'}

OUT = os.path.join(ROOT, 'audit', 'SHA256SUMS.txt')


def _walk_dir(top):
    for dp, dn, fn in os.walk(top):
        dn[:] = [d for d in dn
                 if d not in SKIP_DIRS
                 and not d.startswith(PIP_TEMP_PREFIX)
                 and not d.startswith('.venv')]
        # 跳过不随仓库分发的数据目录（真实语料等）
        rel_dp = os.path.relpath(dp, ROOT).replace(chr(92), '/')
        if any(rel_dp == s or rel_dp.startswith(s + '/') for s in SKIP_DIR_PATHS):
            dn[:] = []
            continue
        for f in sorted(fn):
            if f.endswith(('.pyc', '.pyo')) or f in SKIP_FILES:
                continue
            # 跳过复现会重写的派生结果
            if rel_dp == 'results' and f in SKIP_RESULT_FILES:
                continue
            # 跳过复现会重新导出的文档派生物
            if rel_dp == 'paper' and f in SKIP_PAPER_FILES:
                continue
            yield os.path.join(dp, f)


def _git_tracked():
    """返回 git 索引里的全部文件（仓库相对 POSIX 路径）。
    清单只收 git 跟踪的文件，才能保证「公开仓库里有什么，清单就列什么」。
    """
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        if out.returncode == 0 and out.stdout.strip():
            return set(x for x in out.stdout.split("\n") if x.strip())
    except Exception:
        pass
    return None


def files():
    tracked = _git_tracked()
    if tracked is not None:
        # 首选：遍历 git 索引，天然排除一切未跟踪物
        for rel in sorted(tracked):
            # 白名单目录过滤
            if not any(rel == d or rel.startswith(d + "/") for d in INCLUDE_DIRS):
                if rel not in INCLUDE_ROOT_FILES:
                    continue
            parts = rel.split("/")
            if any(x in SKIP_DIRS for x in parts):
                continue
            if any(rel == s or rel.startswith(s + "/") for s in SKIP_DIR_PATHS):
                continue
            base = parts[-1]
            if base in SKIP_FILES or base.endswith((".pyc", ".pyo")):
                continue
            if len(parts) == 2 and parts[0] == "results" and base in SKIP_RESULT_FILES:
                continue
            if len(parts) == 2 and parts[0] == "paper" and base in SKIP_PAPER_FILES:
                continue
            fp = os.path.join(ROOT, rel.replace("/", os.sep))
            if os.path.isfile(fp):
                yield fp
        return
    # 兜底（无 git 时）：退回磁盘扫描
    for d in INCLUDE_DIRS:
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            for f in _walk_dir(p):
                yield f
    for f in INCLUDE_ROOT_FILES:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            yield p


lines = []
for p in sorted(files()):
    rel = os.path.relpath(p, ROOT).replace(chr(92), '/')
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    lines.append('%s  %s' % (h, rel))

# 自检：任何路径都不该含第三方/系统目录
BAD = ('.venv', 'site-packages', 'node_modules', '%SystemDrive%', 'ProgramData', 'AppData')
bad = [l for l in lines if any(b in l.split('  ', 1)[1] for b in BAD)]
if bad:
    raise SystemExit('[FAIL] 清单仍含非交付物 %d 条，例如: %s' % (len(bad), bad[0][:120]))

header = ('# HCM deliverables SHA256 manifest' + chr(10)
          + '# generated: ' + datetime.now(timezone.utc).replace(microsecond=0).isoformat() + chr(10)
          + '# files: ' + str(len(lines)) + chr(10)
          + '# scope: project deliverables only (src/bench/results/tools/refs/tests/docs/repro/paper/audit/notes + README.md);' + chr(10)
          + '#        third-party virtualenvs and OS caches are excluded by construction' + chr(10)
          + '# 用途：替代缺失的 git 提交哈希。核验方式：' + chr(10)
          + '#   python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],\'rb\').read()).hexdigest())" <file>' + chr(10) + chr(10))

open(OUT, 'w', encoding='utf-8').write(header + chr(10).join(lines) + chr(10))
print('wrote', OUT, 'files=', len(lines))
