'''把 audit/citations-report.json 的核验结果回填进 refs/refs.jsonl。
原则：只覆盖核验产物字段（verified/tier/title_similarity/pmid/pmcid/retracted/is_open_access），
保留 claims/notes 等人工字段；ref_id 为主键；报告里没有的条目原样保留并计入 stale。'''
import json, sys, shutil, os
from datetime import datetime, timezone

REPORT='audit/citations-report.json'; REFS='refs/refs.jsonl'
rep=json.load(open(REPORT,encoding='utf-8'))
recs={r['ref_id']:r for r in rep['records']}
rows=[json.loads(l) for l in open(REFS,encoding='utf-8') if l.strip()]
KEYS=('verified','tier','title_similarity','pmid','pmcid','retracted','is_open_access')
changed=0; stale=[]; missing=[]
for r in rows:
    rid=r.get('ref_id')
    if rid not in recs: stale.append(rid); continue
    src_=recs[rid]
    for k in KEYS:
        if k in src_ and src_[k] is not None and r.get(k)!=src_[k]:
            r[k]=src_[k]; changed+=1
    if src_.get('evidence_level') and r.get('evidence_level') in (None,'','unread'):
        r['evidence_level']=src_['evidence_level']
for rid in recs:
    if rid not in {r.get('ref_id') for r in rows}: missing.append(rid)
shutil.copyfile(REFS, REFS+'.bak')
with open(REFS,'w',encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+chr(10))
print('rows=%d updated_fields=%d stale=%d missing=%d' % (len(rows),changed,len(stale),len(missing)))
if stale: print('stale(no report record):',stale[:5])
if missing: print('missing(not in refs.jsonl):',missing[:5])
from collections import Counter
print('verified:',Counter(r.get('verified') for r in rows))
print('tier:',Counter(r.get('tier') for r in rows))
