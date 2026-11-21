import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, 'memory', '10-STATE.md')
s = open(p, encoding='utf-8').read()

# 顶部加一句终态声明，并把定时器一节改掉
if 'ABANDONED' not in s:
    s = s.replace('# 10-STATE · HCM 记忆模型（断点续跑唯一入口）',
                  '# 10-STATE · HCM 记忆模型（断点续跑唯一入口）' + chr(10) + chr(10)
                  + '> 项目状态：**已交付并冻结**（2026-11-06）。看门狗定时器已移除，不再自动续跑。' + chr(10)
                  + '> 若需继续，读 README.md 的「若继续推进」一节，按 SHA256 清单确认基线后再动手。')
    s = s.replace('- 看门狗 cron：schedule-f38eeff9-d876-4c36-9105-dd9b12811ed3',
                  '- 看门狗 cron：**已于交付后移除**（目标完成后不再需要自动续跑）')
    s = s.replace('- goal：goal-4ebcc554-5674-4a23-a5ac-d8820239eba1',
                  '- goal：goal-4ebcc554-5674-4a23-a5ac-d8820239eba1 —— **phase=complete**（已收尾）')
    open(p, 'w', encoding='utf-8').write(s)
    print('state annotated as final')
else:
    print('already annotated')