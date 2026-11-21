import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'results', 'exp13_confidence.json'), encoding='utf-8'))
p = os.path.join(ROOT, 'paper', 'PAPER.md')
s = open(p, encoding='utf-8').read()

rows = ('| 信号 | corr(·, 正确) | AUC |' + chr(10)
        + '|---|---|---|' + chr(10)
        + '| 融合分 m1 | ' + str(d['corr_m1']) + ' | ' + str(d['auc_m1']) + ' |' + chr(10)
        + '| **raw_evidence（吸引子重建匹配度）** | **' + str(d['corr_raw']) + '** | **' + str(d['auc_raw']) + '** |')

new = ('**必须区分两个信号，这是本节最重要的修正**（bench/exp13_confidence.py）：' + chr(10) + chr(10)
  + rows + chr(10) + chr(10)
  + '**融合分 m1 与正确性几乎无关，甚至略负（AUC ' + str(d['auc_m1']) + '，低于随机）。** '
  + '它把候选排出了序，但这个序不对应答对与否。真正的判别信号是 raw_evidence：库内 '
  + str(d['raw_in_mean']) + '、库外 ' + str(d['raw_out_mean']) + '，AUC = ' + str(d['auc_raw']) + '（完美分离）。' + chr(10) + chr(10)
  + '旧版本曾写 corr(m1,correct)=0.222，该数字**没有任何脚本产出过**（由独立审查发现），'
  + '实测为 ' + str(d['corr_m1']) + '。已删除该数字，改用上表实测值。'
  + '这也意味着：**弃权必须基于 raw_evidence，而不是融合分**，与 3.8 节的设计一致。')

old_head = '曲线单调下降，说明融合分**是有序的**：它把容易的查询排在前面。'
if old_head in s:
    i = s.index(old_head)
    # 找到该段结束（下一个空行后的段落边界）
    j = s.index(chr(10) + chr(10), i)
    s = s[:i] + new + s[j:]
    print('section 4.6 replaced')
else:
    print('WARN: head not found')

s = s.replace('5. **置信度校准的可扩展性**：融合分与正确性的相关（L0 上 0.222）是否随 d_h 上升；若不上升，需要替换为集成式不确定度。',
              '5. **置信度校准的可扩展性**：raw_evidence 的 AUC（L0 上 1.000，样本 120）是否在大 d_h 与大库下保持；若下降，说明重建匹配度会被干扰淹没，需要替换为集成式不确定度。')

# §9.4 那句自陈也改准
s = s.replace('4. **P1-1 的 8/20→20/20 与 r=0.222 无法追溯来源**，已从正文删除并替换为实测值。原始数字保留在此表中作为记录，不进入任何结论。',
              '4. **P1-1 的 8/20→20/20 与 r=0.222 无法追溯来源**，已从正文删除并替换为实测值（exp11 与 exp13）。'
              + chr(10)
              + '   注意：r=0.222 曾在 4.6 与 7 两处残留（声称已删但未删），由第二轮独立审查发现，现已彻底清除；实测值为 corr(m1,correct)=-0.070、AUC=0.458。')

open(p, 'w', encoding='utf-8').write(s)
print('0.222 残留:', s.count('0.222'))
print('chars:', len(s))