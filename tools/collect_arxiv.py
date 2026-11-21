# -*- coding: utf-8 -*-
'''collect_arxiv.py -- 相关工作竞品系统识别类引文采集（amend 2026-11-07）。
role=system：仅用于标识竞争系统，不作为科学论断证据。标题/作者/年份取自 arXiv API 响应本体。
用法：py tools/collect_arxiv.py   （追加去重到 refs/refs.jsonl，ref_id=SYS001..）'''
import json, os, re, sys, time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, 'refs', 'refs.jsonl')

WANTS = [
    ('2310.08560', 'arXiv preprint', 'LLM OS 分页记忆（MemGPT）'),
    ('2304.03442', 'arXiv preprint', '生成式智能体经验流+三因子检索+反思（Generative Agents）'),
    ('2504.19413', 'arXiv preprint', 'LLM CRUD 记忆管线（Mem0）'),
    ('2501.13956', 'arXiv preprint', '双时间轴动态知识图谱（Zep）'),
    ('2405.14831', 'Advances in Neural Information Processing Systems', '海马索引理论+PPR（HippoRAG, NeurIPS 2024）'),
    ('2502.14802', 'arXiv preprint', '非参数持续学习（HippoRAG 2）'),
    ('2502.12110', 'arXiv preprint', 'Zettelkasten 卡片盒记忆（A-MEM）'),
]

def fetch(aid):
    url = 'http://export.arxiv.org/api/query?id_list=%s&max_results=1' % aid
    with urllib.request.urlopen(url, timeout=30) as r:
        xml = r.read().decode('utf-8', 'replace')
    m = re.search(r'<entry>.*?</entry>', xml, re.S)
    if not m: return None
    e = m.group(0)
    tm = re.search(r'<title>(.*?)</title>', e, re.S)
    if not tm: return None
    title = ' '.join(tm.group(1).split())
    authors = re.findall(r'<name>(.*?)</name>', e, re.S)
    pm = re.search(r'<published>(\d{4})', e)
    return dict(title=title, authors=authors, year=int(pm.group(1)) if pm else None)

existing = [json.loads(l) for l in open(REFS, encoding='utf-8') if l.strip()]
have = {r.get('arxiv_id') for r in existing}
next_n = 1 + max([int(r['ref_id'][3:]) for r in existing if r.get('ref_id', '').startswith('SYS')] or [0])
added = []
for aid, journal, note in WANTS:
    if aid in have:
        print('skip (exists):', aid); continue
    rec = fetch(aid)
    if not rec:
        print('[FAIL] no api record for', aid); sys.exit(1)
    fa_parts = rec['authors'][0].split() if rec['authors'] else ['']
    entry = {
        'ref_id': 'SYS%03d' % next_n,
        'module': 'related_work',
        'role': 'system',
        'arxiv_id': aid,
        'doi': None, 'pmid': None, 'pmcid': None,
        'title': rec['title'],
        'first_author': fa_parts[-1] if len(fa_parts) > 1 else rec['authors'][0] if rec['authors'] else '',
        'n_authors': len(rec['authors']),
        'year': rec['year'],
        'journal': journal,
        'tier': 'T2' if 'information processing' in journal.lower() else 'T3',
        'verified': 'pending',
        'title_similarity': None,
        'evidence_level': 'abstract',
        'retracted': False,
        'source_query': 'related-work six-routes survey 2026-11-07',
        'resolved_by': 'pending',
        'retrieved_at': None,
        'claims': [],
        'notes': '〔系统识别〕竞品系统标识，不作为科学论断证据（amend 2026-11-07）: ' + note,
    }
    with open(REFS, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print('added', entry['ref_id'], aid, '|', entry['first_author'], entry['year'], '|', rec['title'][:60])
    added.append(entry['ref_id'])
    next_n += 1
    time.sleep(3.1)  # arXiv API 礼貌间隔
print('done, added', len(added), 'entries')