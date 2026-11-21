import sys, os, json, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import build_paper as bp
SRC = os.path.join(bp.ROOT, 'paper', 'PAPER_EN.md')
OUT = os.path.join(bp.ROOT, 'paper', 'PAPER_EN_final.md')
refs = bp.load_refs()
by_key = {}
for r in refs:
    fa, yr = bp.fold(r.get('first_author') or ''), str(r.get('year') or '')
    if fa: by_key.setdefault((fa, yr), []).append(r)
text = open(SRC, encoding='utf-8').read()
tokens = bp.TOKEN.findall(text)
uniq = []
for t in tokens:
    if t not in uniq: uniq.append(t)
unresolved, chosen = [], {}
for surname, year in uniq:
    k = bp.fold(surname); cands = []
    for (fa, yr), lst in by_key.items():
        if yr == year and (fa in k or k in fa): cands.extend(lst)
    if not cands: unresolved.append(surname + year); continue
    cands.sort(key=lambda r: (bp.RANK.get(r.get('tier'), 3), len(r.get('title') or '')))
    chosen[(surname, year)] = cands[0]
    if len(cands) > 1: print('  [warn] %s%s matched %d' % (surname, year, len(cands)))
if unresolved:
    print('[FAIL] %d unresolved: %s' % (len(unresolved), ', '.join(unresolved))); sys.exit(1)
print('tokens ok: %d unique keys' % len(uniq))
order, numbering = [], {}
for t in tokens:
    if t not in numbering: order.append(t); numbering[t] = len(order)
out = bp.TOKEN.sub(lambda m: '[%d]' % numbering[(m.group(1), m.group(2))], text)
out = out.replace('{{NCITE}}', str(len(refs)))
def fmt_en(num, r):
    fa = r.get('first_author') or ''; n = r.get('n_authors') or 0
    who = (fa + ' et al.') if (n and n > 2) else fa
    title = (r.get('title') or '').strip().rstrip('.')
    line = '[%d] %s (%s). %s. *%s*.' % (num, who, r.get('year'), title, r.get('journal'))
    if r.get('doi'): line += ' doi:' + r['doi']
    if r.get('pmid'): line += ' PMID:' + str(r['pmid'])
    return line + ' [%s]' % r.get('tier')
parts = [out, '---', '## References', '']
for t in order:
    parts.append(fmt_en(numbering[t], chosen[t])); parts.append('')
parts.append('---')
parts.append('## Appendix G: module evidence base (auto-generated)')
parts.append('')
parts.append('The table lists, per model module, the highest-authority evidence sources in the citation bank (at most 8 per module, T1 first). It does NOT claim each reference directly supports a specific claim of this paper; it documents evidence coverage per module. Full fields (DOI, PMID, verification status, evidence level) are in refs/refs.jsonl and audit/citations-report.md.')
parts.append('')
mods = {}
for r in refs: mods.setdefault(r.get('module') or 'UNSORTED', []).append(r)
for m in sorted(mods):
    rows = sorted(mods[m], key=lambda r: (bp.RANK.get(r.get('tier'), 3), -(r.get('year') or 0)))
    parts.append('**%s** (%d records, listing %d)' % (m, len(mods[m]), min(8, len(rows))))
    parts.append('')
    for r in rows[:8]:
        parts.append('- %s (%s). %s. *%s*. [%s]' % (r.get('first_author'), r.get('year'), (r.get('title') or '')[:110], r.get('journal'), r.get('tier')))
    parts.append('')
open(OUT, 'w', encoding='utf-8').write('\n'.join(parts))
print('wrote %s: %d refs, %d chars' % (OUT, len(order), sum(len(p) for p in parts)))