'''Generate paper figures (pure matplotlib, no graphviz).
Default: paper/figures/*.png (150dpi screen).
--plos: journal re-export -> *.tif (600dpi, PLOS width cap 19.05cm=7.5in; oversized fig1/fig2/fig5 narrowed proportionally, font size kept).
Labels are English-only (v1 pre-upload anglicization, 2026-11-09).
'''
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PLOS = '--plos' in sys.argv
DPI = 600 if PLOS else 150
EXT = '.tif' if PLOS else '.png'
# PLOS two-column width cap 7.5in; per-figure width -> scale factor (<=1), height scaled by same ratio, font sizes kept (relatively larger, stays readable)
_WMAX = 7.5
def _fs(w, h):
    if not PLOS or w <= _WMAX:
        return (w, h)
    k = _WMAX / w
    return (round(w * k, 3), round(h * k, 3))

plt.rcParams['font.sans-serif']=['DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=150
OUT='paper/figures'; os.makedirs(OUT,exist_ok=True)
def J(n): return json.load(open('results/'+n,encoding='utf-8'))

# ---------- Fig 1 architecture (nine operators + ablation verdict badges, v2 round 3) ----------
fig,ax=plt.subplots(figsize=_fs(12.6,7.0)); ax.set_xlim(0,13); ax.set_ylim(0,7.8); ax.axis('off')
BADGE={'LB':('LOAD-BEARING','#2e7d32','#e8f5e9'),'RD':('REDUNDANT','#b58900','#fdf6e3'),
       'IC':('INCONCLUSIVE','#616161','#f0f0f0'),'IN':('INERT','#616161','#f0f0f0')}
def box(x,y,w,h,txt,fc='#eef3fb',ec='#3b6ea5',fs=8.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',fc=fc,ec=ec,lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,linespacing=1.45)
def badge(x,y,w,kind):
    lab,ec,fc=BADGE[kind]
    ax.add_patch(FancyBboxPatch((x,y),w,0.34,boxstyle='round,pad=0.02,rounding_size=0.10',fc=fc,ec=ec,lw=1.0))
    ax.text(x+w/2,y+0.17,lab,ha='center',va='center',fontsize=6.6,color=ec,fontweight='bold')
def arr(x1,y1,x2,y2,txt='',color='#444',ls='-',fs=7.2,dy=0.14,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.15,color=color,ls=ls,
                                 connectionstyle='arc3,rad=%g'%rad))
    if txt:
        ax.text((x1+x2)/2,(y1+y2)/2+dy,txt,ha='center',fontsize=fs,color=color)
# --- row 1: encoding ---
box(0.2,5.7,2.1,1.15,'Sensory input\n$x_t$ + context $c_t$\n(timestamp / turn)','#fafafa','#888')
box(2.9,5.7,3.7,1.15,'M1_SEP pattern separation (EC·DG)\nexpectation-library residual $u=\\mathrm{unit}(x-\\mu_j)$\nexpansion + k-WTA: $h=\\mathrm{kWTW}(s,k)$','#fdf3e3','#c8873a')
badge(3.25,5.28,1.5,'LB')
box(7.2,5.7,2.7,1.15,'M2_EPI episodic store (CA3)\none-shot write $M\\leftarrow[M,\\,g\\cdot h]$\nHopfield: $a=\\mathrm{softmax}(\\beta M^\\top q)$','#e9f0fa','#2f5d94')
badge(7.9,5.28,1.3,'LB')
box(10.4,5.7,2.4,1.15,'M3_CTX contextual bias\n(EC→CA1 modulation)\n$\\kappa\\,\\langle c,W_c\\rangle$','#f6f6f6','#999')
badge(11.0,5.28,1.2,'RD')
arr(2.3,6.27,2.9,6.27)
arr(6.6,6.27,7.2,6.27,'','#2f5d94',dy=0)
ax.text(6.72,5.98,'sparse code $h$',ha='center',fontsize=7.0,color='#2f5d94')
ax.text(6.5,0.22,'Dual readout: same $W_e$ projection — sparse code supports, dense evidence drives retrieval (§5.2)',ha='center',fontsize=7.4,style='italic',color='#555')
# --- row 2: modulation and semantics ---
box(0.2,3.35,2.6,1.15,'M6_WM working memory (PFC)\n$C$ θ-γ slots\ncapacity knee $\\approx C$','#f2ecfa','#6a4fa3')
badge(0.85,2.93,1.3,'LB')
box(3.2,3.35,3.0,1.15,'M7_NEU neuromodulatory gating\n$g=\\sigma(w\\cdot[\\nu,r,\\varepsilon,a]+b)$\nVTA / LC / amygdala','#fbecec','#a5443f')
badge(3.9,2.93,1.6,'IC')
box(6.75,3.35,6.05,1.15,'M5_SEM semantic store (neocortex)  $f(x)=B\\,\\tanh(Ax)$\nper-sample delta + cascaded plasticity (Benna–Fusi): fast-then-slow\nretrieval prior $s_n=\\sigma(z(f))$, fused with episodic attractor $f=\\lambda a+(1-\\lambda)s_n$','#eef7f8','#2c7d84')
badge(9.0,2.93,1.3,'LB')
# --- row 3: control and slow processes ---
box(0.2,1.0,2.6,1.15,'M0_META meta-control (PFC)\nrecall / separate / abstain\nevidence: $d\\prime$=7.12, AUC=1.0','#f5f5f5','#555')
badge(0.85,0.58,1.3,'LB')
box(3.2,1.0,3.0,1.15,'M4_CON consolidation (SWR replay)\ninterleaved replay → old tasks 0→16/24\n(priority vs uniform: no difference)','#f3f0e6','#8a7a3a')
badge(4.0,0.58,1.3,'LB')
box(6.45,1.0,6.35,1.15,'M8_FOR forgetting (PFC–hippocampus)\nimportance-based pruning · retrieval-induced suppression (RIF) · usage decay\nin current architecture: suppression spreads store-wide, top-1 unchanged (INERT)','#f0f0f0','#666')
badge(9.0,0.58,1.2,'IN')
# --- flows ---
arr(1.5,5.7,1.5,4.5,'',color='#6a4fa3',ls='--',dy=0)
ax.text(1.62,5.06,'item buffer',fontsize=6.8,color='#6a4fa3')
arr(5.9,3.35,5.9,5.7,'','#a5443f',ls='--',dy=0)
arr(7.5,5.7,7.5,4.92,'',color='#2f5d94',dy=0)
ax.text(7.3,5.06,'$a$ attractor',fontsize=6.8,color='#2f5d94',ha='right')
arr(10.2,4.5,10.2,5.7,'',color='#2c7d84',dy=0)
ax.text(10.35,5.06,'$s_n$ prior',fontsize=6.8,color='#2c7d84')
ax.plot([6.65,6.65],[2.15,5.30],color='#666',ls=':',lw=1.0)
ax.add_patch(FancyArrowPatch((6.65,5.30),(7.65,5.64),arrowstyle='-|>',mutation_scale=10,lw=1.0,color='#666',ls=':',connectionstyle='arc3,rad=0.25'))
ax.text(6.44,3.75,'RIF suppression',fontsize=6.8,color='#666',rotation=90,va='center')
ax.text(5.78,4.95,'write gate $g\\cdot h$',fontsize=6.8,color='#a5443f',ha='right')
ax.text(9.25,4.82,'$f=\\lambda a+(1-\\lambda)s_n$',fontsize=7.0,color='#444',ha='center')
arr(1.5,3.35,1.5,2.15,'',color='#555',ls=':',dy=0)
ax.text(1.62,2.6,'readout',fontsize=6.8,color='#555')
arr(4.7,2.15,4.7,3.35,'',color='#8a7a3a',ls='--',dy=0)
ax.text(4.84,2.62,'replay write (interleaved)',fontsize=6.8,color='#8a7a3a')
arr(7.6,2.15,7.6,3.35,'',color='#666',ls=':',dy=0)
ax.text(7.74,2.62,'pruning / decay',fontsize=6.8,color='#666')
ax.text(6.5,7.52,'The nine HCM operators: hippocampal–cortical complementary memory',ha='center',fontsize=13,fontweight='bold')
ax.text(6.5,7.12,'Badge = ablation verdict (§4.8): 6 LOAD-BEARING / 1 REDUNDANT (M3) / 1 INCONCLUSIVE (M7) / 1 INERT (M8); scale tiers L0 laptop → L3 multi-node (§7)',ha='center',fontsize=7.6,color='#444')
plt.tight_layout(); plt.savefig(OUT+'/fig1_architecture'+EXT,dpi=DPI,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'} ); plt.close()
# ---------- Fig 2 separation ----------
d = J('exp1_separation.json')
d = d['rows'] if isinstance(d, dict) else d
ovs = sorted({r['overlap'] for r in d})
fig, axes = plt.subplots(1, 2, figsize=_fs(9.2, 3.6))
for resid, st, col in ((False, 'o--', '#c0392b'), (True, 's-', '#1f6f3f')):
    lab = 'with residual' if resid else 'no residual'
    ys = [next((r['n0'] for r in d if r['overlap'] == o and r['resid'] == resid and r['L'] == 3), 0) for o in ovs]
    axes[0].plot(ovs, ys, st, color=col, label=lab)
    cs = [next((r['code_cos'] for r in d if r['overlap'] == o and r['resid'] == resid and r['L'] == 0), 0) for o in ovs]
    axes[1].plot(ovs, cs, st, color=col, label=lab)
axes[0].set_xlabel('input overlap'); axes[0].set_ylabel('exact recall top-1 / 30')
axes[0].set_title('(a) Recall under high overlap'); axes[0].legend(frameon=False); axes[0].grid(alpha=.3)
axes[0].set_ylim(-1, 31)
axes[1].set_xlabel('input overlap'); axes[1].set_ylabel('inter-item code cosine')
axes[1].set_title('(b) Code separability'); axes[1].legend(frameon=False); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + '/fig2_separation'+EXT, dpi=DPI, bbox_inches='tight', pil_kwargs={'compression':'tiff_lzw'}); plt.close()

# ---------- Fig 3 capacity ----------
e=J('exp2c_capacity_edge.json')
Ns=sorted({r['N'] for r in e})
fig,ax=plt.subplots(figsize=_fs(6.6,3.6))
for resid,st,col in ((False,'o--','#c0392b'),(True,'s-','#1f6f3f')):
    ax.plot(Ns,[next(r['degraded'] for r in e if r['N']==N and r['resid']==resid) for N in Ns],st,color=col,label=('with residual' if resid else 'no residual'))
ax.axhline(20,color='#888',ls=':',lw=1,label='exact cue (perfect recall = 20)')
ax.set_xscale('log'); ax.set_xlabel('Items in store (N)'); ax.set_ylabel('25%-degraded cue top-1')
ax.set_title('Capacity boundary: the bottleneck is denoising, not item count'); ax.legend(frameon=False); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+'/fig3_capacity'+EXT,dpi=DPI,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'} ); plt.close()

# ---------- Fig 4 continual learning ----------
c=J('exp6_continual.json')
fig,ax=plt.subplots(figsize=_fs(5.6,3.4))
arms=['blocked','uniform','priority']; lab={'blocked':'no replay','uniform':'interleaved (uniform)','priority':'interleaved (priority)'}
x=np.arange(len(arms)); A=[r['A'] for r in c]; B=[r['B'] for r in c]
ax.bar(x-0.19,A,0.36,label='old tasks 0–3',color='#c0392b')
ax.bar(x+0.19,B,0.36,label='new tasks 4–7',color='#2f6f9f')
ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in arms],fontsize=8.5)
ax.set_ylabel('held-out accuracy (out of 24)'); ax.set_ylim(0,26); ax.legend(frameon=False,fontsize=8)
ax.set_title('Interleaved replay rescues old tasks from collapse')
plt.tight_layout(); plt.savefig(OUT+'/fig4_replay'+EXT,dpi=DPI,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'} ); plt.close()

# ---------- Fig 5 working memory + confidence ----------
w=J('exp7_wm.json')
fig,axes=plt.subplots(1,2,figsize=_fs(9.2,3.4))
Cs=[r['C'] for r in w]; K=[r['kept'] for r in w]
axes[0].plot(Cs,K,'o-',color='#6a4fa3',label='retained items (measured)')
axes[0].plot([1,8],[1,8],'--',color='#888',label='ideal capacity')
axes[0].set_xlabel('slots C (load L=8)'); axes[0].set_ylabel('retained items')
axes[0].set_title('(a) Working-memory capacity'); axes[0].legend(frameon=False); axes[0].grid(alpha=.3)
meta=J('exp8_meta.json'); cur=meta['curve']
axes[1].plot([r['coverage'] for r in cur],[r['risk'] for r in cur],'o-',color='#a5443f')
axes[1].set_xlabel('coverage (answer rate)'); axes[1].set_ylabel('risk (error rate among answers)')
axes[1].set_title('(b) Risk–coverage curve with usable confidence'); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT+'/fig5_wm_confidence'+EXT,dpi=DPI,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'} ); plt.close()

print('figures:',sorted(os.listdir(OUT)))
