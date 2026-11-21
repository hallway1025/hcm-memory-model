'''G7：单命令全流程复现（一把梭）。
从零跑：全部基准 → 图表 → 论文编译 → 三向比对。任一步失败即非零退出。'''
import subprocess, sys, os, time

# --- 控制台编码兜底：中文 Windows 默认 GBK，子进程输出含替换字符时会崩 ---
try:
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# ---

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

STEPS = [
    ('分离/完成主表', 'bench\\exp1_separation.py'),
    ('容量边界', 'bench\\exp2c_capacity_edge.py'),
    ('持续学习', 'bench\\exp6_continual.py'),
    ('受限回放', 'bench\\exp6b_replay_buffer.py'),
    ('工作记忆容量', 'bench\\exp7_wm_capacity.py'),
    ('元认知/弃权', 'bench\\exp8_meta.py'),
    ('情境绑定(严格)', 'bench\\exp9_context_strict.py'),
    ('双读出对照', 'bench\\exp11_readout.py'),
    ('生成器敏感性', 'bench\\exp12_generator_sensitivity.py'),
    ('置信度信号', 'bench\\exp13_confidence.py'),
    ('priority 功效(20种子)', 'bench\\exp15_priority_power.py'),
    ('多种子统计', 'bench\\exp10_stats.py'),
    ('统一消融', 'bench\\ablation.py'),
    ('专属消融', 'bench\\ablation2.py'),
    ('韧度扫描', 'bench\\durability.py'),
    ('G4 总账', 'bench\\g4_ledger.py'),
    ('E1 互补性(D-052漂移面)', 'bench\\exp1_complementarity.py'),
    ('exp36b κ标定(~25min)', 'bench\\exp36b_kappa_calib.py'),
    ('微观偏置冒烟', 'bench\\_micro_ebias.py'),
    ('图表', 'tools\\make_figures.py'),
    ('论文编译', 'tools\\build_paper.py'),
    ('三向比对', 'tools\\check_claims.py'),
    ('导出 DOCX', 'tools\\export_docx.py'),
    ('导出 HTML', 'tools\\export_html.py'),
    # 未入清单：bench/exp17_sparse_index.py（2026-11-08 工单#06 记录冲突，不硬加）。
    # 冲突原因：其 main() 硬编码快跑参数（SEEDS=[0], NS=[2000]），而
    # results/exp17_sparse_index.json 是全量产物（N=[2000,16000] x seeds[0,1,2]
    # 共 6 rows），被 paper 附录 D 引用；按现 main() 重跑会用快跑产物覆盖全量
    # 工件，构成论文引用降级面。入清单前须先在脚本内恢复全量默认（或加
    # --quick 开关区分产物），否则宁可手工跑。
]

fail = []
t0 = time.time()
for name, rel in STEPS:
    path = os.path.join(ROOT, rel)
    t = time.time()
    print('[%s] %s ...' % (time.strftime('%H:%M:%S'), name), flush=True)
    r = subprocess.run([PY, '-u', path], cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    ok = r.returncode == 0
    print('    %s  %.1fs' % ('OK' if ok else 'FAIL(%d)' % r.returncode, time.time() - t), flush=True)
    if not ok:
        fail.append((name, rel, (r.stderr or r.stdout)[-1200:]))

print()
print('=' * 60)
if fail:
    print('FAILED %d / %d steps' % (len(fail), len(STEPS)))
    for name, rel, err in fail:
        print('\n--- %s (%s) ---' % (name, rel))
        print(err)
    sys.exit(1)
print('ALL %d STEPS PASSED in %.1fs' % (len(STEPS), time.time() - t0))