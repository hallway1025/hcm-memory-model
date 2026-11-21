#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp29_text_frontend.py - F2 公平真实语料对照（文本前端四臂）+ 统计工具（F3 复用）。

阶段（--stage）:
  selftest  判据/统计函数单测 + 小型真实 HCM 集成自检（不写语料）
  smoke     peer-01 前 1000、四臂、gate200、deg0.3（管线验证）
  diag      peer-01 前 2000、四臂、gate200+emoji20+marker20、deg 0/0.3/0.5、d'（B1 依据）
  full      全 train 拟合 + test 留出（intrain200/dup200/rand200）；--arm 单臂运行

臂（固定顺序 NEW256 -> R256 -> S256npz -> R512）:
  NEW256   src/hcm/text_encoder.py CharacterSemanticEncoder(256,window=3,seed=0)，
           只在阶段 train split fit（F1 审查 N-2 要求）
  R256     字符随机投影 dim=256（exp22p/exp23e 语义：rng(7) 列归一 W）
  S256npz  results/exp23_semantic_embed.npz 旧 S256（泄漏参考臂；fit 仅统计覆盖率，
           查询/写入对词表外字符用 blake2b 确定性回退，exp23e FB(99) 语义扩展）
  R512     字符随机投影 dim=512（exp22p 原口径）

共同超参: Config(d_x=d, d_c=d, a=0.02, capacity=N+10, beta=100.0, L=3,
                use_resid=False, lam=1.0, seed=0)；全部 query(do_rif=False)。
维度变化时 d_h=d_x*8、k=round(a*d_h) 随之同步（手册 L-029）。

判据（手册 L-035）: hit = action in {recall,separate} 且 idx>=0 且 obs_of[slot]==target_obs，
obs_of[slot]=int(E.t[slot])（写入时的观察序号）。拒写=miss，分母固定=探针数（手册：
不得按各臂已写集合筛探针）。内容等价口径: 返回槽位 payload（E.P 列）与目标输入向量
余弦>=0.999（与精确事件口径分列）。
探针与损坏位置在写入前由语料文本唯一决定（本文件内冻结规则 + JSON 记录删除位置表）。
d' = (mean_self-mean_other)/sqrt((var_self+var_other)/2)，样本=rng(1234) 冻结的他码池。

结果: results/exp29_text_frontend.json（分 key 增量合并）。探针记录只存索引/删除位置/
SHA256，不存语料正文；报告不得出现聊天正文（手册 §7.3）。
"""
import argparse
import contextlib
import csv
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcm import HCM, Config                      # noqa: E402
from hcm.ops import unit                         # noqa: E402
from hcm.text_encoder import CharacterSemanticEncoder  # noqa: E402

RESULT_PATH = os.environ.get("EXP29_OUT") or os.path.join(ROOT, "results", "exp29_text_frontend.json")
NPZ_PATH = os.path.join(ROOT, "results", "exp23_semantic_embed.npz")

TRAIN_FILES = ["train__peer-02.csv", "train__group-02.csv",
"train__group-01.csv", "train__group-04.csv",
"train__group-03.csv", "train__group-05.csv"]  # v2 六会话（D-053）
TEST_FILES = ["test__peer-02.csv", "test__group-02.csv",
"test__group-01.csv", "test__group-04.csv",
"test__group-03.csv", "test__group-05.csv"]  # v2 六会话

SEEDS = {"proj": 7, "deg_base": 41000, "dprime": 1234, "pairs": 5678, "nnz": 5679,
         "intrain": 2029, "randtest": 3049, "boot": 7}
BETA, L, LAM = 100.0, 3, 1.0
ARM_ORDER = ["NEW256", "R256", "S256npz", "R512"]
LEVEL_KEY = {0.3: "del03", 0.5: "del05"}

NANO = re.compile(r"\[[^\]]{1,8}\]")
EMOJI_RULE = ("U+1F000-1FAFF or U+2600-27BF or U+2B00-2BFF or U+1F1E6-1F1FF "
              "or {U+FE0F, U+200D}")
CONTENT_COS_THR = 0.999


def clean_text(t):
    """与 exp22p/exp23e/text_encoder 完全一致的清洗：去 [..] 标记 + 去全部空白。"""
    return NANO.sub("", re.sub(r"\s+", "", str(t)))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_texts(fname, limit=None):
    path = os.path.join(ROOT, "bench", "corpus", fname)
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            out.append(r[2] if len(r) > 2 else "")
            if limit is not None and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- io / env

def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_data(path, data):
    data = sanitize(data)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=1)
    os.replace(tmp, path)


def merge_key(path, key, value):
    data = load_json(path)
    data[key] = value
    save_data(path, data)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def ram_info():
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return {"mem_load_pct": int(st.dwMemoryLoad),
                "total_mb": int(st.ullTotalPhys // (1024 * 1024)),
                "free_mb": int(st.ullAvailPhys // (1024 * 1024))}
    return {}


def peak_ws_bytes():
    """进程累计峰值工作集（单调不减；跨阶段值不可比，仅记录绝对峰值，N-3 口径注明）。"""
    try:
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    except Exception:
        pass
    return None


def np_config_capture():
    try:
        cfg = getattr(np, "__config__", None)
        d = getattr(cfg, "CONFIG", None)
        if isinstance(d, dict):
            return {"type": "np.__config__.CONFIG",
                    "value": json.loads(json.dumps(d, default=str))}
    except Exception:
        pass
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            np.show_config()
        return {"type": "np.show_config", "value": buf.getvalue()[:4000]}
    except Exception as e:
        return {"type": "error", "value": str(e)}


def stage_meta():
    meta = {"created_at": now_iso(), "python": sys.version.split()[0],
            "numpy": np.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "script_sha256": None,
            "git_head": None}
    try:
        meta["script_sha256"] = sha256_file(os.path.abspath(__file__))
    except Exception:
        pass
    try:
        meta["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=20).stdout.strip() or None
    except Exception as e:
        meta["git_head"] = "unavailable: %s" % e
    sha = {}
    for fn in TRAIN_FILES + TEST_FILES:
        try:
            sha[fn] = sha256_file(os.path.join(ROOT, "bench", "corpus", fn))
        except Exception as e:
            sha[fn] = "error: %s" % e
    try:
        meta["npz_exp23_sha256"] = sha256_file(NPZ_PATH)
    except Exception:
        meta["npz_exp23_sha256"] = None
    meta["corpus_sha256"] = sha
    meta["seeds"] = SEEDS
    meta["np_config"] = np_config_capture()
    meta["ram_at_start"] = ram_info()
    meta["hyperparams"] = {"beta": BETA, "L": L, "lam": LAM, "use_resid": False,
                           "do_rif": False, "a": 0.02, "capacity": "N+10",
                           "d_c": "d_x", "d_h": "d_x*8", "k": "round(a*d_h)",
                           "seed_model": 0}
    return meta


# ------------------------------------------------------------ probes（冻结）

def has_emoji(s):
    for ch in s:
        o = ord(ch)
        if (0x1F000 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or \
           (0x2B00 <= o <= 0x2BFF) or (0x1F1E6 <= o <= 0x1F1FF) or o in (0xFE0F, 0x200D):
            return True
    return False


def iso_scores(cleaned):
    chars = sorted({c for t in cleaned if t for c in t})
    vidx = {c: j for j, c in enumerate(chars)}
    B = np.zeros((len(cleaned), len(chars)), dtype=np.float32)
    for i, t in enumerate(cleaned):
        for c in set(t):
            B[i, vidx[c]] = 1.0
    cnt = B @ B.T
    sz = B.sum(axis=1)
    contain = cnt / np.maximum(sz, 1.0)[:, None]
    np.fill_diagonal(contain, 0.0)
    return contain.max(axis=1)


def corrupt_positions(cleaned, frac, seed):
    """与 exp22p encode_deg 同构：fresh default_rng(seed)，choice 删除位置（置''）。"""
    r = np.random.default_rng(seed)
    n = len(cleaned)
    if n > 1:
        return sorted(int(j) for j in r.choice(n, size=max(1, int(n * frac)), replace=False))
    return []


def apply_del(cleaned, del_pos):
    ds = set(del_pos)
    return "".join(c for j, c in enumerate(cleaned) if j not in ds)


def build_stage1_probes(texts, cleaned, n_gate=200, n_sub=20):
    n = len(texts)
    iso = iso_scores(cleaned)
    body = [i for i in range(n) if cleaned[i]]
    specs = [("gate", [i for i in body if iso[i] < 0.5][:n_gate]),
             ("emoji20", [i for i in body if has_emoji(cleaned[i])][:n_sub]),
             ("marker20", [i for i in range(n) if not cleaned[i]][:n_sub])]
    groups = {}
    for gname, srcs in specs:
        recs = []
        for j, i in enumerate(srcs):
            recs.append({"pid": j, "qsrc": i, "target": i,
                         "del03": corrupt_positions(cleaned[i], 0.3, SEEDS["deg_base"] + j),
                         "del05": corrupt_positions(cleaned[i], 0.5, SEEDS["deg_base"] + j),
                         "clean_sha": sha256_text(cleaned[i]),
                         "len_clean": len(cleaned[i])})
        groups[gname] = recs
    rng = np.random.default_rng(SEEDS["dprime"])
    pool = {}
    for gname, _ in specs:
        lst = []
        for rec in groups[gname]:
            cand = rng.choice(n, size=51, replace=False).tolist()
            lst.append([c for c in cand if c != rec["qsrc"]][:50])
        pool[gname] = lst
    return {"rule": ("gate=cleaned nonempty & iso<0.5, first 200 by index (no written filter); "
                     "emoji20=body with emoji, first 20; marker20=cleaned empty, first 20; "
                     "deg seed=41000+pid; qsrc/target=corpus row index (0-based)"),
            "iso_rule": "max_{j!=i} |C_i&cap;C_j|/|C_i| over cleaned char sets",
            "emoji_rule": EMOJI_RULE,
            "n_corpus": n, "n_body": len(body),
            "iso_lt_half_body_count": int(sum(1 for i in body if iso[i] < 0.5)),
            "frozen_at": now_iso(),
            "groups": groups, "dprime_others": pool}


def build_full_probes(train_raw, train_clean, test_raw, test_clean):
    n_tr, n_te = len(train_raw), len(test_raw)
    body = [i for i in range(n_tr) if train_clean[i]]
    rng = np.random.default_rng(SEEDS["intrain"])
    pick = rng.choice(len(body), size=200, replace=False)
    intrain = []
    for j, k in enumerate(pick):
        i = int(body[int(k)])
        intrain.append({"pid": j, "qsrc": i, "target": i,
                        "del03": corrupt_positions(train_clean[i], 0.3, SEEDS["deg_base"] + j),
                        "del05": corrupt_positions(train_clean[i], 0.5, SEEDS["deg_base"] + j),
                        "clean_sha": sha256_text(train_clean[i]),
                        "len_clean": len(train_clean[i])})
    first_train = {}
    for i, c in enumerate(train_clean):
        if c and c not in first_train:
            first_train[c] = i
    dup = []
    for c in test_clean:
        if c and c in first_train:
            i = first_train[c]
            dup.append({"pid": len(dup), "qsrc": None, "target": i,
                        "del03": corrupt_positions(c, 0.3, SEEDS["deg_base"] + len(dup)),
                        "test_clean_sha": sha256_text(c),
                        "train_clean_sha": sha256_text(train_clean[i])})
            if len(dup) >= 200:
                break
    # dup200 需要测试侧行号：重扫一遍带索引
    dup = []
    for j, c in enumerate(test_clean):
        if c and c in first_train:
            i = first_train[c]
            dup.append({"pid": len(dup), "qsrc": j, "target": i,
                        "del03": corrupt_positions(c, 0.3, SEEDS["deg_base"] + len(dup)),
                        "test_clean_sha": sha256_text(c),
                        "train_clean_sha": sha256_text(train_clean[i])})
            if len(dup) >= 200:
                break
    body_te = [i for i in range(n_te) if test_clean[i]]
    rng2 = np.random.default_rng(SEEDS["randtest"])
    rpick = rng2.choice(len(body_te), size=200, replace=False)
    vocab = {c for t in train_clean for c in t}
    rand = []
    for j, k in enumerate(rpick):
        i = int(body_te[int(k)])
        c = test_clean[i]
        rand.append({"pid": j, "qsrc": i, "target": None,
                     "del03": corrupt_positions(c, 0.3, SEEDS["deg_base"] + j),
                     "clean_sha": sha256_text(c),
                     "len_clean": len(c),
                     "oov_chars": len({ch for ch in c} - vocab),
                     "has_emoji": has_emoji(c)})
    return {"rule": ("intrain200=rng(2029) body texts deg 0/0.3/0.5 (target=own obs); "
                     "dup200=first 200 test texts whose cleaned text equals some train text "
                     "(target=train first-occurrence obs, deg 0/0.3, exact-event + content口径); "
                     "rand200=rng(3049) body test texts deg 0/0.3 (no ground truth: action dist "
                     "+ payload source identification + char containment + OOV subgroup); "
                     "deg seed=41000+pid; qsrc=corpus row index (sessions concatenated in order "
                     "peer-01/_/水泊/滚木); sha=cleaned text SHA256 (no text stored)"),
            "n_train": n_tr, "n_test": n_te,
            "train_vocab_chars": len(vocab),
            "n_dup_pool_available": sum(1 for c in test_clean if c and c in first_train),
            "frozen_at": now_iso(),
            "groups": {"intrain200": intrain, "dup200": dup, "rand200": rand}}


# ------------------------------------------------------------------- arms

class RArm:
    """字符随机投影臂（exp22p/exp23e 语义）。查询 OOV 字符用 blake2b 确定性回退。"""

    def __init__(self, dim, vocab, seed=SEEDS["proj"]):
        self.name = "R%d" % dim
        self.dim = int(dim)
        self.seed = int(seed)
        self.vocab = list(vocab)
        self.VOC = {c: j for j, c in enumerate(self.vocab)}
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((self.dim, len(self.vocab)), dtype=np.float32)
        self.W /= np.linalg.norm(self.W, axis=0, keepdims=True) + 1e-9
        self._fb = {}

    def fb_vec(self, ch):
        v = self._fb.get(ch)
        if v is None:
            key = ("fb:%d:%s" % (self.seed, ch)).encode("utf-8")
            s = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            v = unit(np.random.default_rng(s).standard_normal(self.dim).astype(np.float32))
            self._fb[ch] = v
        return v

    def chars_vec(self, chars):
        parts = []
        for c in chars:
            j = self.VOC.get(c)
            parts.append(self.W[:, j] if j is not None else self.fb_vec(c))
        if not parts:
            parts.append(self.W[:, self.VOC["\ufffd"]])
        return unit(np.sum(parts, axis=0, dtype=np.float32))

    def encode_inputs(self, raw, cleaned):
        return np.asarray([self.chars_vec(list(c) or ["\ufffd"]) for c in cleaned],
                          dtype=np.float32)

    def encode_query(self, qstr, raw):
        chars = list(clean_text(qstr if qstr else raw)) or ["\ufffd"]
        return self.chars_vec(chars)


class S256NpzArm:
    """旧 S256（exp23 npz，泄漏参考臂）。fit 仅统计覆盖率；OOV 用 FB(99) 语义回退。"""

    def __init__(self, npz_path, fb_seed=99):
        z = np.load(npz_path, allow_pickle=False)
        self.chars = [str(c) for c in z["chars"]]
        self.E = np.asarray(z["E"], dtype=np.float32)   # (Vn, dim)，行向量
        self.dim = int(self.E.shape[1])
        self.VOC = {c: j for j, c in enumerate(self.chars)}
        self.fb_seed = int(fb_seed)
        self._fb = {}
        self.name = "S256npz"

    def coverage_report(self, cleaned):
        seen = {c for t in cleaned for c in t}
        in_v = sum(1 for c in seen if c in self.VOC)
        return {"vocab_npz": len(self.chars), "stage_chars_seen": len(seen),
                "stage_chars_in_vocab": int(in_v),
                "stage_chars_oov": int(len(seen) - in_v),
                "ufffd_in_npz_vocab": "\ufffd" in self.VOC}

    def fb_vec(self, ch):
        v = self._fb.get(ch)
        if v is None:
            key = ("fb:%d:%s" % (self.fb_seed, ch)).encode("utf-8")
            s = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            v = unit(np.random.default_rng(s).standard_normal(self.dim).astype(np.float32))
            self._fb[ch] = v
        return v

    def chars_vec(self, chars):
        parts = []
        for c in chars:
            j = self.VOC.get(c)
            parts.append(self.E[j] if j is not None else self.fb_vec(c))
        if not parts:
            if "\ufffd" in self.VOC:
                parts.append(self.E[self.VOC["\ufffd"]])
            else:
                parts.append(self.fb_vec("\ufffd"))
        return unit(np.sum(parts, axis=0, dtype=np.float32))

    def encode_inputs(self, raw, cleaned):
        return np.asarray([self.chars_vec(list(c)) for c in cleaned], dtype=np.float32)

    def encode_query(self, qstr, raw):
        chars = list(clean_text(qstr if qstr else raw))
        return self.chars_vec(chars)


class NEW256Arm:
    """新语义臂：src/hcm/text_encoder.py，只在阶段 train split fit（N-2）。
    写入传 RAW 文本（编码器内部 normalize）；查询传损坏后 cleaned 串，空则传 RAW
    以触发 empty:+原文 身份回退（与各自写路径一致）。"""

    def __init__(self, dim=256, window=3, seed=0):
        self.enc = CharacterSemanticEncoder(dim=dim, window=window, seed=seed)
        self.dim = int(dim)
        self.name = "NEW256"

    def fit(self, raw_texts):
        return self.enc.fit(raw_texts)

    def encode_inputs(self, raw, cleaned):
        return np.asarray(self.enc.transform(raw), dtype=np.float32)

    def encode_query(self, qstr, raw):
        return self.enc.encode(qstr if qstr else raw)


def make_arm(name, train_raw, train_clean):
    if name in ("R512", "R256"):
        dim = 512 if name == "R512" else 256
        vocab = sorted({c for t in train_clean for c in t} | {"\ufffd"})
        return RArm(dim, vocab, seed=SEEDS["proj"]), {"vocab_size": len(vocab),
                                                      "proj_seed": SEEDS["proj"]}
    if name == "S256npz":
        return S256NpzArm(NPZ_PATH, fb_seed=99), {"npz": os.path.relpath(NPZ_PATH, ROOT)}
    if name == "NEW256":
        return NEW256Arm(dim=256, window=3, seed=0), {"new_fit": "stage train split raw texts",
                                                      "encoder": "src/hcm/text_encoder.py"}
    raise ValueError("unknown arm: %s" % name)


# ------------------------------------------------------------- HCM 运行/评测

def write_all(arm, X, n):
    m = HCM(Config(d_x=arm.dim, d_c=arm.dim, a=0.02, capacity=n + 10, beta=BETA, L=L,
                   use_resid=False, lam=LAM, seed=0))
    wrote = gate_rej = dup_rej = 0
    t0 = time.perf_counter()
    for i in range(n):
        rec = m.observe(X[i])
        if rec["wrote"]:
            wrote += 1
        elif rec["g"] >= m.cfg.theta_write:
            dup_rej += 1
        else:
            gate_rej += 1
        if (i + 1) % 20000 == 0:
            print("    write %d/%d wrote=%d dup=%d gate=%d %.0fs"
                  % (i + 1, n, wrote, dup_rej, gate_rej, time.perf_counter() - t0),
                  flush=True)
    dt = time.perf_counter() - t0
    obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
    slot_of_obs = {v: s for s, v in obs_of.items()}
    wstats = {"wrote": wrote, "gate_rejected": gate_rej, "dup_or_full_rejected": dup_rej}
    return m, obs_of, slot_of_obs, dt, wstats


def judge(action, idx, obs_of, target_obs):
    """L-035 正确判据。abstain / idx<0 / 槽位观察序号不等于目标 -> miss。"""
    if action not in ("recall", "separate"):
        return False
    if idx is None:
        return False
    idx = int(idx)
    if idx < 0:
        return False
    return obs_of.get(idx, -1) == int(target_obs)


def content_eq(pv, xv):
    return bool(float(np.dot(pv, xv)) >= CONTENT_COS_THR)


def dprime_stats(self_cos, other_cos):
    a = np.asarray(self_cos, dtype=np.float64)
    b = np.asarray(other_cos, dtype=np.float64)
    out = {"n_self": int(a.size), "n_other": int(b.size), "mean_self": None,
           "var_self": None, "mean_other": None, "var_other": None, "dprime": None,
           "rule": "d'=(m_self-m_other)/sqrt((v_self+v_other)/2), ddof=1, cos deg0.3 query vs written input vecs"}
    if a.size >= 2 and b.size >= 2:
        ms, mo = float(a.mean()), float(b.mean())
        vs, vo = float(a.var(ddof=1)), float(b.var(ddof=1))
        dp = (ms - mo) / math.sqrt((vs + vo) / 2.0) if (vs + vo) > 0 else None
        out.update(mean_self=ms, var_self=vs, mean_other=mo, var_other=vo, dprime=dp)
    return out


def evaluate_group(m, arm, qtexts, qclean, recs, level, obs_of, slot_of_obs,
                   Xref, others_pool=None, want_dprime=False):
    hits = content = abstain = 0
    actions = {}
    self_cos, other_cos = [], []
    q_time = 0.0
    per_probe = []
    for j, rec in enumerate(recs):
        qs = rec["qsrc"]
        if level <= 0.0:
            qstr = qclean[qs]
        else:
            qstr = apply_del(qclean[qs], rec[LEVEL_KEY[level]])
        t0 = time.perf_counter()
        q = arm.encode_query(qstr, qtexts[qs])
        o = m.query(q, do_rif=False)
        q_time += time.perf_counter() - t0
        act = str(o.get("action"))
        actions[act] = actions.get(act, 0) + 1
        if act == "abstain":
            abstain += 1
        idx = o.get("idx")
        target = rec.get("target")
        ok = target is not None and judge(act, idx, obs_of, target)
        if ok:
            hits += 1
        cok = False
        if (target is not None and act in ("recall", "separate")
                and idx is not None and int(idx) >= 0 and target in slot_of_obs):
            cok = content_eq(m.E.P[:, int(idx)], Xref[target])
        if cok:
            content += 1
        per_probe.append({"pid": rec["pid"], "hit": bool(ok), "content": bool(cok),
                          "action": act})
        if want_dprime and others_pool is not None:
            if target in slot_of_obs:
                self_cos.append(float(np.dot(q, Xref[target])))
            for oi in others_pool[j]:
                if oi in slot_of_obs and oi != target:
                    other_cos.append(float(np.dot(q, Xref[oi])))
    out = {"n": len(recs), "hit": hits, "content_hit": content, "abstain": abstain,
           "actions": actions, "query_time_s": round(q_time, 3)}
    if want_dprime:
        out["dprime"] = dprime_stats(self_cos, other_cos)
    return out, per_probe


def rand_eval(m, arm, test_raw, test_clean, recs, level, id_map, train_clean):
    acts = {}
    abstain = 0
    identified = 0
    contain_sum = 0.0
    contain_n = 0
    oov_acts = {}
    q_time = 0.0
    for rec in recs:
        qs = rec["qsrc"]
        qstr = test_clean[qs] if level <= 0.0 else apply_del(test_clean[qs], rec["del03"])
        t0 = time.perf_counter()
        q = arm.encode_query(qstr, test_raw[qs])
        o = m.query(q, do_rif=False)
        q_time += time.perf_counter() - t0
        act = str(o.get("action"))
        acts[act] = acts.get(act, 0) + 1
        if act == "abstain":
            abstain += 1
        if rec.get("oov_chars", 0) > 0:
            oov_acts[act] = oov_acts.get(act, 0) + 1
        idx = o.get("idx")
        if act in ("recall", "separate") and idx is not None and int(idx) >= 0:
            pv = m.E.P[:, int(idx)]
            src = id_map.get(hashlib.md5(np.asarray(pv, dtype=np.float32).tobytes()).digest())
            if src is not None:
                identified += 1
                sp = set(train_clean[src])
                pp = set(test_clean[qs])
                contain_sum += len(sp & pp) / max(1, len(pp))
                contain_n += 1
    n_oov = sum(1 for r in recs if r.get("oov_chars", 0) > 0)
    return {"n": len(recs), "actions": acts, "abstain": abstain,
            "src_identified": identified,
            "char_contain_mean": (contain_sum / contain_n) if contain_n else None,
            "n_oov_probes": n_oov, "oov_subgroup_actions": oov_acts,
            "query_time_s": round(q_time, 3),
            "rule": "source=payload(E.P col) md5->written input row; contain=|C_probe&cap;C_src|/|C_probe|"}


def code_metrics(m, X, slot_of_obs, n_pairs=2000, nnz_sample=10000):
    n_slots = int(m.E.n)
    out = {"n_slots": n_slots, "n_pairs": 0, "code_cos_mean": None,
           "code_cos_absmean": None, "nnz_code_mean": None,
           "input_rand_cos_mean": None, "nnz_input_mean": None}
    if n_slots < 2:
        return out
    M = m.E.M
    rng = np.random.default_rng(SEEDS["pairs"])
    s1 = rng.integers(0, n_slots, size=n_pairs)
    s2 = rng.integers(0, n_slots, size=n_pairs)
    keep = s1 != s2
    s1, s2 = s1[keep], s2[keep]
    if s1.size:
        c1 = M[:, s1]
        c2 = M[:, s2]
        den = np.linalg.norm(c1, axis=0) * np.linalg.norm(c2, axis=0)
        den = np.where(den <= 0, 1e-12, den)
        cos = np.einsum("ij,ij->j", c1, c2) / den
        out["n_pairs"] = int(cos.size)
        out["code_cos_mean"] = float(cos.mean())
        out["code_cos_absmean"] = float(np.abs(cos).mean())
    tot = 0
    for start in range(0, n_slots, 2048):
        end = min(n_slots, start + 2048)
        tot += int((np.abs(M[:, start:end]) > 1e-8).sum())
    out["nnz_code_mean"] = tot / n_slots
    wobs = np.array(sorted(slot_of_obs.keys()), dtype=np.int64)
    rng2 = np.random.default_rng(SEEDS["pairs"])
    if wobs.size >= 2:
        k = min(n_pairs, wobs.size)
        o1 = rng2.choice(wobs, size=k, replace=wobs.size < n_pairs)
        o2 = rng2.choice(wobs, size=k, replace=wobs.size < n_pairs)
        keep = o1 != o2
        o1, o2 = o1[keep], o2[keep]
        if o1.size:
            out["input_rand_cos_mean"] = float(
                np.einsum("ij,ij->j", X[o1], X[o2]).mean())
        samp = rng2.choice(wobs, size=min(nnz_sample, wobs.size), replace=False)
        out["nnz_input_mean"] = float((np.abs(X[samp]) > 1e-8).sum(axis=1).mean())
    return out


# ------------------------------------------------------------ stage 1 (smoke/diag)

def run_one_arm_stage1(name, texts, cleaned, probes, levels):
    arm, info = make_arm(name, texts, cleaned)
    fit_dt = 0.0
    new_vocab = None
    if name == "NEW256":
        t0 = time.perf_counter()
        arm.fit(texts)
        fit_dt = time.perf_counter() - t0
        new_vocab = len(arm.enc.vocab)
    t0 = time.perf_counter()
    X = arm.encode_inputs(texts, cleaned)
    enc_dt = time.perf_counter() - t0
    m, obs_of, slot_of_obs, write_dt, wstats = write_all(arm, X, len(texts))
    res = {"dim": arm.dim, "d_h": arm.dim * 8, "k": int(round(0.02 * arm.dim * 8)),
           "n_messages": len(texts), "n_written": int(m.E.n),
           "reject_rate": round(1.0 - int(m.E.n) / max(1, len(texts)), 6),
           "write_stats": wstats, "fit_time_s": round(fit_dt, 3),
           "encode_time_s": round(enc_dt, 3), "write_time_s": round(write_dt, 3)}
    res.update(info)
    if new_vocab is not None:
        res["new_vocab"] = new_vocab
    if name == "S256npz":
        res["coverage"] = arm.coverage_report(cleaned)
    groups_out = {}
    per_probe = {}
    qtot = 0.0
    for gname in ("gate", "emoji20", "marker20"):
        recs = probes["groups"][gname]
        per_probe[gname] = {}
        groups_out[gname] = {"n": len(recs)}
        for lv in levels:
            r, pp = evaluate_group(m, arm, texts, cleaned, recs, lv, obs_of, slot_of_obs,
                                   X, others_pool=probes["dprime_others"][gname],
                                   want_dprime=(gname == "gate"))
            per_probe[gname][lv] = pp
            qtot += r["query_time_s"]
            groups_out[gname]["%.1f" % lv] = {k: v for k, v in r.items() if k != "dprime"}
            if "dprime" in r:
                groups_out[gname]["dprime@%.1f" % lv] = r["dprime"]
    gw = {rec["qsrc"] for rec in probes["groups"]["gate"] if rec["qsrc"] in slot_of_obs}
    res["gate_written_srcs"] = sorted(gw)
    res["probe_coverage"] = {
        "gate": len(gw),
        "emoji20": sum(1 for rec in probes["groups"]["emoji20"] if rec["qsrc"] in slot_of_obs),
        "marker20": sum(1 for rec in probes["groups"]["marker20"] if rec["qsrc"] in slot_of_obs)}
    res["groups"] = groups_out
    res["query_time_s_total"] = round(qtot, 3)
    res["code_metrics"] = code_metrics(m, X, slot_of_obs)
    pws = peak_ws_bytes()
    res["peak_ws_mb"] = round(pws / 1e6, 1) if pws else None
    res["completed_at"] = now_iso()
    return res, gw, per_probe


def probes_public(probes):
    return probes


def run_stage1(stage_name, n_first, levels):
    texts = read_texts(TRAIN_FILES[0], limit=n_first)
    cleaned = [clean_text(t) for t in texts]
    probes = build_stage1_probes(texts, cleaned)
    arms_res = {}
    gate_written = {}
    per_probe_store = {}
    t_stage = time.perf_counter()
    for name in ARM_ORDER:
        print("[%s] arm %s start" % (stage_name, name), flush=True)
        try:
            res, gw, pp = run_one_arm_stage1(name, texts, cleaned, probes, levels)
        except Exception:
            res = {"error": traceback.format_exc()}
            gw, pp = set(), None
        arms_res[name] = res
        gate_written[name] = gw
        per_probe_store[name] = pp
        gc.collect()
        print("[%s] arm %s done" % (stage_name, name), flush=True)
    ok_arms = [n for n in ARM_ORDER if "error" not in arms_res[n]]
    common = None
    for n in ok_arms:
        s = gate_written[n]
        common = s if common is None else (common & s)
    common = common or set()
    gate_recs = probes["groups"]["gate"]
    common_out = {"n_common": len(common),
                  "coverage": round(len(common) / max(1, len(gate_recs)), 4),
                  "arms": {}}
    for n in ok_arms:
        lvl_out = {}
        for lv in levels:
            pp = per_probe_store[n]["gate"][lv]
            h = sum(1 for rec, r in zip(gate_recs, pp) if rec["qsrc"] in common and r["hit"])
            c = sum(1 for rec, r in zip(gate_recs, pp) if rec["qsrc"] in common and r["content"])
            lvl_out["%.1f" % lv] = {"hit": h, "content_hit": c, "n": len(common)}
        common_out["arms"][n] = lvl_out
    out = {"probes": probes_public(probes), "arms": arms_res,
           "gate_common_written": common_out,
           "stage_time_s": round(time.perf_counter() - t_stage, 1)}
    merge_key(RESULT_PATH, stage_name, out)
    for n in ARM_ORDER:
        r = arms_res[n]
        if "error" in r:
            print("[%s] %s ERROR (see json)" % (stage_name, n), flush=True)
            continue
        g = r["groups"]["gate"]
        line = ["[%s] %s written=%d/%d" % (stage_name, n, r["n_written"], r["n_messages"])]
        for lv in levels:
            e = g["%.1f" % lv]
            line.append("deg%s=%d/%d(content %d)" % (lv, e["hit"], e["n"], e["content_hit"]))
        if "dprime@%.1f" % 0.3 in g:
            line.append("d'0.3=%s" % (("%.3f" % g["dprime@0.3"]["dprime"])
                                      if g["dprime@0.3"]["dprime"] is not None else "NA"))
        print(" ".join(line), flush=True)
    print("[%s] gate_common_written n=%d" % (stage_name, common_out["n_common"]), flush=True)
    print("[%s] STAGE_OK" % stage_name, flush=True)


# ---------------------------------------------------------------- full 阶段

def run_one_arm_full(name, train_raw, train_clean, test_raw, test_clean, probes):
    arm, info = make_arm(name, train_raw, train_clean)
    fit_dt = 0.0
    if name == "NEW256":
        t0 = time.perf_counter()
        arm.fit(train_raw)
        fit_dt = time.perf_counter() - t0
        info["new_vocab"] = len(arm.enc.vocab)
    t0 = time.perf_counter()
    X = arm.encode_inputs(train_raw, train_clean)
    enc_dt = time.perf_counter() - t0
    m, obs_of, slot_of_obs, write_dt, wstats = write_all(arm, X, len(train_raw))
    res = {"dim": arm.dim, "d_h": arm.dim * 8, "k": int(round(0.02 * arm.dim * 8)),
           "n_train": len(train_raw), "n_test": len(test_raw),
           "n_written": int(m.E.n),
           "reject_rate": round(1.0 - int(m.E.n) / max(1, len(train_raw)), 6),
           "write_stats": wstats, "fit_time_s": round(fit_dt, 3),
           "encode_time_s": round(enc_dt, 3), "write_time_s": round(write_dt, 3)}
    res.update(info)
    if name == "S256npz":
        res["coverage"] = arm.coverage_report(train_clean)
    groups_out = {}
    qtot = 0.0
    gin = {}
    for lv in (0.0, 0.3, 0.5):
        r, _ = evaluate_group(m, arm, train_raw, train_clean, probes["groups"]["intrain200"],
                              lv, obs_of, slot_of_obs, X)
        qtot += r["query_time_s"]
        gin["%.1f" % lv] = r
    groups_out["intrain200"] = gin
    gdup = {}
    for lv in (0.0, 0.3):
        r, _ = evaluate_group(m, arm, test_raw, test_clean, probes["groups"]["dup200"],
                              lv, obs_of, slot_of_obs, X)
        qtot += r["query_time_s"]
        gdup["%.1f" % lv] = r
    groups_out["dup200"] = gdup
    id_map = {}
    for o in slot_of_obs:
        id_map[hashlib.md5(np.asarray(X[o], dtype=np.float32).tobytes()).digest()] = o
    grand = {}
    for lv in (0.0, 0.3):
        r = rand_eval(m, arm, test_raw, test_clean, probes["groups"]["rand200"],
                      lv, id_map, train_clean)
        qtot += r["query_time_s"]
        grand["%.1f" % lv] = r
    groups_out["rand200"] = grand
    res["groups"] = groups_out
    res["probe_coverage"] = {
        "intrain_written": sum(1 for rec in probes["groups"]["intrain200"]
                               if rec["qsrc"] in slot_of_obs),
        "dup_targets_written": sum(1 for rec in probes["groups"]["dup200"]
                                   if rec["target"] in slot_of_obs)}
    res["query_time_s_total"] = round(qtot, 3)
    res["code_metrics"] = code_metrics(m, X, slot_of_obs)
    pws = peak_ws_bytes()
    res["peak_ws_mb"] = round(pws / 1e6, 1) if pws else None
    res["completed_at"] = now_iso()
    del m, X, arm, id_map
    gc.collect()
    return res


def run_full(only_arm=None):
    train_raw = []
    for fn in TRAIN_FILES:
        train_raw.extend(read_texts(fn))
    train_clean = [clean_text(t) for t in train_raw]
    test_raw = []
    for fn in TEST_FILES:
        test_raw.extend(read_texts(fn))
    test_clean = [clean_text(t) for t in test_raw]
    data = load_json(RESULT_PATH)
    full = data.get("full") or {}
    if "probes" not in full:
        probes = build_full_probes(train_raw, train_clean, test_raw, test_clean)
        full["probes"] = probes
        full["arms"] = {}
        data["full"] = full
        save_data(RESULT_PATH, data)
        print("[full] probes frozen at %s" % probes["frozen_at"], flush=True)
    else:
        probes = full["probes"]
        full["arms"] = full.get("arms") or {}
        data["full"] = full
        print("[full] probes reused (frozen %s)" % probes.get("frozen_at"), flush=True)
    names = [only_arm] if only_arm else list(ARM_ORDER)
    for name in names:
        prev = full["arms"].get(name)
        if prev and "error" not in prev:
            print("[full] arm %s already done, skip" % name, flush=True)
            continue
        print("[full] arm %s start" % name, flush=True)
        try:
            res = run_one_arm_full(name, train_raw, train_clean, test_raw, test_clean, probes)
        except Exception:
            res = {"error": traceback.format_exc()}
        data = load_json(RESULT_PATH)
        data.setdefault("full", {}).setdefault("arms", {})[name] = res
        save_data(RESULT_PATH, data)
        gc.collect()
        if "error" in res:
            print("[full] arm %s ERROR (see json)" % name, flush=True)
        else:
            print("[full] arm %s done written=%d/%d intrain0=%d/200 dup0=%d/200(exact)"
                  % (name, res["n_written"], res["n_train"],
                     res["groups"]["intrain200"]["0.0"]["hit"],
                     res["groups"]["dup200"]["0.0"]["hit"]), flush=True)
    print("[full] STAGE_OK", flush=True)


# ------------------------------------------------------------- F3 语义收益验收

F3_SYN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_synpairs.json")
F3_PARA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_paraphrases.json")
F3_ARM_ORDER = ["NEW256", "R256"]
F3_HC_SEEDS = [0, 1, 2]
# Lead 放行附加要求①：按 zero / low / high 三层分层（zero=overlap==0）
F3_STRATA = [("zero", 0.0, 1e-9), ("low", 1e-9, 0.5), ("high", 0.5, 1.01)]
# 冻结设计决定（在任何 F3 结果产生之前）：改写检索使用 peer-01-only store。
# peer-01 是四会话拼接顺序的第一个会话，故"目标在拼接语料中的首现观察序号"
# 与 peer-01-only store 内的观察序号完全一致；编码器仍按 N-2 在全 train fit。
F3_STORE_NOTE = ("paraphrase store = train__peer-01.csv only (n_david messages, "
                 "capacity=n+10); encoder fit = full 4-session train (N-2); "
                 "target obs = first occurrence in peer-01 == first occurrence in "
                 "concatenated order (peer-01 is first session)")


def _f3_cos(a, b):
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def _f3_verify_assets(para, syn):
    ok = True
    for q in para["queries"]:
        pass  # 目标正文不在冻结文件中，sha 校验在 run_f3 载入语料后进行
    for doc, name in ((syn, "synpairs"), (para, "paraphrases")):
        body = json.dumps({k: v for k, v in doc.items() if k != "sha256"},
                          ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if h != doc.get("sha256"):
            print("[f3] ASSET SHA MISMATCH: %s" % name, flush=True)
            ok = False
    return ok


def _f3_agg_queries(qres):
    n = len(qres)
    ex = sum(1 for r in qres if r["exact_hit"])
    ce = sum(1 for r in qres if r["content_hit"])
    abst = sum(1 for r in qres if r["action"] == "abstain")
    acts = {}
    for r in qres:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    tw = sum(1 for r in qres if r["target_written"])
    sims = [r["target_sim"] for r in qres if r["target_sim"] is not None]
    strat = {}
    for sname, lo, hi in F3_STRATA:
        sub = [r for r in qres if lo <= r["overlap"] < hi]
        strat[sname] = {"n": len(sub),
                        "exact": sum(1 for r in sub if r["exact_hit"]),
                        "content": sum(1 for r in sub if r["content_hit"])}
    return {"n_queries": n, "targets_written": tw, "exact_hit": ex,
            "content_hit": ce, "abstain": abst, "actions": acts,
            "mean_target_sim": round(float(np.mean(sims)), 6) if sims else None,
            "strata": strat}


def run_f3():
    syn = load_json(F3_SYN_PATH)
    para = load_json(F3_PARA_PATH)
    if not _f3_verify_assets(para, syn):
        raise SystemExit("frozen asset sha mismatch - abort")
    david_raw = read_texts(TRAIN_FILES[0])  # v2: peer-02（peer-01 别名，首会话不变）
    david_clean = [clean_text(t) for t in david_raw]
    train_raw = []
    for fn in TRAIN_FILES:
        train_raw.extend(read_texts(fn))
    train_clean = [clean_text(t) for t in train_raw]
    n_david = len(david_raw)

    # 目标定位 + 冻结 sha 校验（sha 对不上即中止：语料或冻结文件被改动）
    targets = {}
    for q in para["queries"]:
        r = int(q["row"])
        tclean = david_clean[r]
        h = hashlib.sha256(tclean.encode("utf-8")).hexdigest()[:16]
        if h != q["target_clean_sha"]:
            raise SystemExit("target sha mismatch at row %d (pid %s)"
                             % (r, q["pid"]))
        fo = david_clean.index(tclean)   # 首现观察序号
        targets[q["pid"]] = {"row": r, "first_obs": fo}
    fo_mismatch = sum(1 for q in para["queries"]
                      if targets[q["pid"]]["first_obs"] != int(q["row"]))
    print("[f3] targets verified: %d, first_obs!=row count=%d"
          % (len(targets), fo_mismatch), flush=True)

    all_pairs = list(syn["pairs"]) + list(syn["unrelated_pairs"])
    arms_out = {}
    qdetail = {}    # name -> seed -> per-query records
    pdetail = {}    # name -> seed -> per-pair code records
    for name in F3_ARM_ORDER:
        print("[f3] arm %s start" % name, flush=True)
        t0 = time.perf_counter()
        arm, info = make_arm(name, train_raw, train_clean)
        fit_dt = 0.0
        if name == "NEW256":
            tf = time.perf_counter()
            arm.fit(train_raw)
            fit_dt = time.perf_counter() - tf
            info["new_vocab"] = len(arm.enc.vocab)
        X = arm.encode_inputs(david_raw, david_clean)

        # 同义/无关对：输入余弦（编码器固定，与 HCM seed 无关）
        pair_in = []
        for pr in all_pairs:
            va = arm.encode_query(pr["a"], pr["a"])
            vb = arm.encode_query(pr["b"], pr["b"])
            pair_in.append({"pid": pr["pid"], "kind": pr["kind"],
                            "zero_shared": pr.get("zero_shared"),
                            "input_cos": _f3_cos(va, vb)})

        seeds_out = {}
        for hc_seed in F3_HC_SEEDS:
            m = HCM(Config(d_x=arm.dim, d_c=arm.dim, a=0.02,
                           capacity=n_david + 10, beta=BETA, L=L,
                           use_resid=False, lam=LAM, seed=hc_seed))
            wrote = 0
            for i in range(n_david):
                rec = m.observe(X[i])
                if rec["wrote"]:
                    wrote += 1
            obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
            slot_of_obs = {v: s for s, v in obs_of.items()}

            # 对的 HCM 码余弦：每对独立小 HCM(capacity=10)，先写 a 再写 b
            pair_code = []
            for pr in all_pairs:
                va = arm.encode_query(pr["a"], pr["a"])
                vb = arm.encode_query(pr["b"], pr["b"])
                mm = HCM(Config(d_x=arm.dim, d_c=arm.dim, a=0.02, capacity=10,
                                beta=BETA, L=L, use_resid=False, lam=LAM,
                                seed=hc_seed))
                mm.observe(va)
                mm.observe(vb)
                c2 = int(mm.E.n) >= 2
                cc = _f3_cos(mm.E.M[:, 0], mm.E.M[:, 1]) if c2 else None
                pair_code.append({"pid": pr["pid"], "kind": pr["kind"],
                                  "wrote_both": c2, "code_cos": cc})
                del mm

            # 改写检索（peer-01-only store）
            qres = []
            for q in para["queries"]:
                fo = targets[q["pid"]]["first_obs"]
                tgt_slot = slot_of_obs.get(fo)
                qv = arm.encode_query(q["query"], q["query"])
                o = m.query(qv, do_rif=False)
                idx = int(o["idx"])
                ex_hit = False
                ce = False
                tsim = None
                if tgt_slot is not None:
                    ex_hit = bool(judge(o["action"], idx, obs_of, fo))
                    pv = np.asarray(m.E.P[:, idx], dtype=np.float32) if idx >= 0 else None
                    ce = bool(pv is not None and content_eq(pv, X[fo]))
                    tsim = _f3_cos(qv, X[fo])
                qres.append({"pid": q["pid"], "overlap": q["char_overlap_ratio"],
                             "exact_hit": ex_hit, "content_hit": ce,
                             "action": str(o["action"]), "target_sim": tsim,
                             "target_written": tgt_slot is not None})
            seeds_out[str(hc_seed)] = {"n_written": int(m.E.n),
                                       "query_detail": qres,
                                       "pair_code_detail": pair_code}
            seeds_out[str(hc_seed)]["query_agg"] = _f3_agg_queries(qres)
            del m
            print("[f3] arm %s seed %d done (%.0fs)"
                  % (name, hc_seed, time.perf_counter() - t0), flush=True)

        # 同义对输入余弦聚合
        def _mean_kind(recs, kind, key):
            v = [r[key] for r in recs if r["kind"] == kind and r[key] is not None]
            return round(float(np.mean(v)), 6) if v else None
        syn_in = _mean_kind(pair_in, "synonym", "input_cos")
        unr_in = _mean_kind(pair_in, "unrelated", "input_cos")
        code_by_seed = {}
        for hcs, sd in seeds_out.items():
            pd = sd["pair_code_detail"]
            code_by_seed[hcs] = {"syn_code_mean": _mean_kind(pd, "synonym", "code_cos"),
                                 "unrel_code_mean": _mean_kind(pd, "unrelated", "code_cos")}
        qdetail[name] = {h: sd["query_detail"] for h, sd in seeds_out.items()}
        pdetail[name] = {h: sd["pair_code_detail"] for h, sd in seeds_out.items()}
        arms_out[name] = {"dim": arm.dim, "info": info, "fit_time_s": round(fit_dt, 3),
                          "n_david": n_david, "n_written_max_seed":
                              max(sd["n_written"] for sd in seeds_out.values()),
                          "pair_input_detail": pair_in,
                          "syn_input_mean": syn_in, "unrel_input_mean": unr_in,
                          "code_means_by_seed": code_by_seed,
                          "seeds": {h: {"n_written": sd["n_written"],
                                        "query_agg": sd["query_agg"]}
                                    for h, sd in seeds_out.items()},
                          "query_detail": qdetail[name],
                          "pair_code_detail": pdetail[name],
                          "elapsed_s": round(time.perf_counter() - t0, 1)}
        del arm, X
        gc.collect()

    # NEW256 vs R256 逐查询配对比较（按 pid 对齐；逐 seed + 跨 seed 合并）
    def _by_pid(detail):
        return {r["pid"]: r for r in detail}

    def _paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        pt = paired_t(d)
        bs = bootstrap_ci_mean(d)
        return {"n": pt["n"], "mean_diff": pt["mean"], "sd": pt["sd"],
                "df": pt["df"], "t": pt["t"], "p_two_sided": pt["p_two"],
                "ci95": pt["ci95"],
                "bootstrap_ci95_mean_diff": [round(v, 6) for v in bs]}

    comparisons = {"query_paired_NEW_vs_R256": {}, "pair_input_paired_NEW_vs_R256": {},
                   "pair_code_paired_NEW_vs_R256": {}}
    seed_keys = [str(x) for x in F3_HC_SEEDS]
    for hcs in seed_keys + ["pooled"]:
        if hcs == "pooled":
            qn = {}; qr = {}
            for h in seed_keys:
                qn.update(_by_pid(qdetail["NEW256"][h]))
                qr.update(_by_pid(qdetail["R256"][h]))
            # pooled 时同 pid 跨 seed 重复出现——保留（配对差按 (seed,pid) 计）
            qn = {}; qr = {}
            for h in seed_keys:
                for r in qdetail["NEW256"][h]:
                    qn[(h, r["pid"])] = r
                for r in qdetail["R256"][h]:
                    qr[(h, r["pid"])] = r
        else:
            qn = _by_pid(qdetail["NEW256"][hcs])
            qr = _by_pid(qdetail["R256"][hcs])
        keys = sorted(set(qn) & set(qr))
        def _val(dd, k, f):
            v = dd[k][f]
            return 0.0 if v is None else float(v)
        comparisons["query_paired_NEW_vs_R256"][hcs] = {
            "exact_hit(1-0)": _paired([1.0 if qn[k]["exact_hit"] else 0.0 for k in keys],
                                      [1.0 if qr[k]["exact_hit"] else 0.0 for k in keys]),
            "content_hit(1-0)": _paired([1.0 if qn[k]["content_hit"] else 0.0 for k in keys],
                                        [1.0 if qr[k]["content_hit"] else 0.0 for k in keys]),
            "target_sim": _paired([_val(qn, k, "target_sim") for k in keys],
                                  [_val(qr, k, "target_sim") for k in keys])}
    # 低重叠子集（语义收益关键证据）：overlap < 0.33 的配对差，逐 seed + pooled
    strat_cmp = {}
    for hcs in seed_keys + ["pooled"]:
        if hcs == "pooled":
            qn = {}; qr = {}
            for h in seed_keys:
                for r in qdetail["NEW256"][h]:
                    qn[(h, r["pid"])] = r
                for r in qdetail["R256"][h]:
                    qr[(h, r["pid"])] = r
        else:
            qn = _by_pid(qdetail["NEW256"][hcs])
            qr = _by_pid(qdetail["R256"][hcs])
        keys = [k for k in sorted(set(qn) & set(qr)) if qn[k]["overlap"] < 0.33]
        if keys:
            strat_cmp[hcs] = {"n_low_overlap": len(keys),
                "exact_hit(1-0)": _paired([1.0 if qn[k]["exact_hit"] else 0.0 for k in keys],
                                          [1.0 if qr[k]["exact_hit"] else 0.0 for k in keys]),
                "target_sim": _paired([qn[k]["target_sim"] or 0.0 for k in keys],
                                      [qr[k]["target_sim"] or 0.0 for k in keys])}
    comparisons["low_overlap_subset_NEW_vs_R256"] = strat_cmp
    pin = {"NEW256": _by_pid(arms_out["NEW256"]["pair_input_detail"]),
           "R256": _by_pid(arms_out["R256"]["pair_input_detail"])}
    # Lead 放行附加要求②：14 个零共享近义对 NEW256 vs R256 逐对胜负表（B1 核心证据，n=14 小样本）
    zero_pids = sorted(p for p in pin["NEW256"]
                       if pin["NEW256"][p]["kind"] == "synonym"
                       and pin["NEW256"][p].get("zero_shared") and p in pin["R256"])
    head_to_head = []
    for p in zero_pids:
        cn = pin["NEW256"][p]["input_cos"]
        cr = pin["R256"][p]["input_cos"]
        row = {"pid": p, "input_cos_NEW256": round(cn, 6),
               "input_cos_R256": round(cr, 6),
               "winner_input": "NEW256" if cn > cr else ("R256" if cr > cn else "tie")}
        ccn = []; ccr = []
        for hcs in seed_keys:
            a = pdetail["NEW256"][hcs]; b = pdetail["R256"][hcs]
            an = {r["pid"]: r for r in a}.get(p); bn = {r["pid"]: r for r in b}.get(p)
            if an and bn and an["code_cos"] is not None and bn["code_cos"] is not None:
                ccn.append(an["code_cos"]); ccr.append(bn["code_cos"])
        if ccn:
            mn, mr = float(np.mean(ccn)), float(np.mean(ccr))
            row["code_cos_mean_NEW256"] = round(mn, 6)
            row["code_cos_mean_R256"] = round(mr, 6)
            row["winner_code"] = "NEW256" if mn > mr else ("R256" if mr > mn else "tie")
        head_to_head.append(row)
    comparisons["zero_shared_synpair_head_to_head"] = {
        "n_pairs": len(head_to_head), "note": "n=14 small sample, report as-is",
        "rows": head_to_head,
        "summary": {
            "NEW_wins_input": sum(1 for r in head_to_head if r["winner_input"] == "NEW256"),
            "R256_wins_input": sum(1 for r in head_to_head if r["winner_input"] == "R256"),
            "NEW_wins_code": sum(1 for r in head_to_head if r.get("winner_code") == "NEW256"),
            "R256_wins_code": sum(1 for r in head_to_head if r.get("winner_code") == "R256")}}
    # 同义/无关对：输入余弦与码余弦的 NEW vs R256 配对差（按 pid）
    for kind in ("synonym", "unrelated"):
        pids = sorted(p for p in pin["NEW256"]
                      if pin["NEW256"][p]["kind"] == kind and p in pin["R256"])
        comparisons["pair_input_paired_NEW_vs_R256"][kind] = _paired(
            [pin["NEW256"][p]["input_cos"] for p in pids],
            [pin["R256"][p]["input_cos"] for p in pids])
    for hcs in seed_keys:
        cn = _by_pid(pdetail["NEW256"][hcs])
        cr = _by_pid(pdetail["R256"][hcs])
        for kind in ("synonym", "unrelated"):
            pids = sorted(p for p in cn if cn[p]["kind"] == kind and p in cr
                          and cn[p]["code_cos"] is not None and cr[p]["code_cos"] is not None)
            comparisons["pair_code_paired_NEW_vs_R256"].setdefault(kind, {})[hcs] = _paired(
                [cn[p]["code_cos"] for p in pids], [cr[p]["code_cos"] for p in pids])
    # ---- post-hoc 廉价基线（Lead 评审后追加；不改冻结资产；与预注册臂分列） ----
    # 预测已在 results/optimization_f2_f3_report.md §4 登记（commit 2504db6，先于本代码运行）：
    # 零重叠层基线对目标相似度构造性为 0，hit@1 应≈0；若命中 => 隔离性异常 => 停止并报告 Lead。
    print("[f3] post-hoc baselines start", flush=True)
    tb0 = time.perf_counter()
    from collections import Counter

    def _bigrams(t):
        return [t[i:i + 2] for i in range(len(t) - 1)]

    def _ngset(t, n):
        return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()

    def _set_cos(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / math.sqrt(len(a) * len(b))

    n_docs = len(david_clean)
    doc_sets2 = [_ngset(t, 2) for t in david_clean]
    doc_sets3 = [_ngset(t, 3) for t in david_clean]
    doc_tf = [Counter(_bigrams(t)) for t in david_clean]
    doc_len = [max(1, len(t) - 1) for t in david_clean]   # bigram token 数（含重复）
    avgdl = sum(doc_len) / float(n_docs)
    df2 = Counter()
    for ds in doc_sets2:
        for g in ds:
            df2[g] += 1
    K1, B = 1.5, 0.75
    idf = {g: math.log(1.0 + (n_docs - dfv + 0.5) / (dfv + 0.5))
           for g, dfv in df2.items()}

    def bm25_score(qterms, tfm, dl):
        sc = 0.0
        dl_norm = 1.0 - B + B * dl / avgdl
        for g in qterms:
            tf = tfm.get(g, 0)
            if tf:
                sc += idf.get(g, 0.0) * tf * (K1 + 1.0) / (tf + K1 * dl_norm)
        return sc

    def _rank_stats(scores, ti):
        st = scores[ti]
        greater = 0
        for v in scores:
            if v > st:
                greater += 1
        rank = greater + 1
        hit1 = bool(greater == 0 and st > 0.0)
        hit5 = bool(greater < 5 and st > 0.0)
        return rank, hit1, hit5, float(st)

    base_q = []
    n2_hit_z = n3_hit_z = bm_hit_z = 0
    for q in para["queries"]:
        ti = targets[q["pid"]]["first_obs"]
        qcl = q["query_clean"]
        q2 = _ngset(qcl, 2)
        q3 = _ngset(qcl, 3)
        qterms = set(_bigrams(qcl))
        row = {"pid": q["pid"], "overlap": q["char_overlap_ratio"]}
        for nm, qs, dsets in (("ngram2", q2, doc_sets2), ("ngram3", q3, doc_sets3)):
            sc = [_set_cos(qs, ds) for ds in dsets]
            rk, h1, h5, st = _rank_stats(sc, ti)
            row[nm + "_rank"] = rk
            row[nm + "_hit1"] = h1
            row[nm + "_hit5"] = h5
            row[nm + "_target_score"] = round(st, 6)
        scb = [bm25_score(qterms, doc_tf[di], doc_len[di]) for di in range(n_docs)]
        rk, h1, h5, st = _rank_stats(scb, ti)
        row["bm25_rank"] = rk
        row["bm25_hit1"] = h1
        row["bm25_hit5"] = h5
        row["bm25_target_score"] = round(st, 6)
        if abs(row["overlap"]) < 1e-9:
            n2_hit_z += 1 if row["ngram2_hit1"] else 0
            n3_hit_z += 1 if row["ngram3_hit1"] else 0
            bm_hit_z += 1 if row["bm25_hit1"] else 0
        base_q.append(row)

    base_pair = []
    for pr in all_pairs:
        a2, b2 = _ngset(pr["a"], 2), _ngset(pr["b"], 2)
        a3, b3 = _ngset(pr["a"], 3), _ngset(pr["b"], 3)
        ta, tb = set(_bigrams(pr["a"])), set(_bigrams(pr["b"]))
        tfa = Counter(_bigrams(pr["a"]))
        tfb = Counter(_bigrams(pr["b"]))
        la = max(1, len(pr["a"]) - 1)
        lb = max(1, len(pr["b"]) - 1)
        bm_pair = 0.5 * (bm25_score(ta, tfb, lb) + bm25_score(tb, tfa, la))
        base_pair.append({"pid": pr["pid"], "kind": pr["kind"],
                          "zero_shared": pr.get("zero_shared"),
                          "ngram2_cos": round(_set_cos(a2, b2), 6),
                          "ngram3_cos": round(_set_cos(a3, b3), 6),
                          "bm25_sym": round(bm_pair, 6)})

    def _bagg(strata_rec, key):
        out_s = {}
        for sname, lo, hi in F3_STRATA:
            sub = [r for r in strata_rec if lo <= r["overlap"] < hi] if key == "overlap" else strata_rec
            out_s[sname if key == "overlap" else "all"] = {
                "n": len(sub),
                "hit1": sum(1 for r in sub if r.get(key + "_hit1")),
                "hit5": sum(1 for r in sub if r.get(key + "_hit5")),
                "mean_target_score": round(float(np.mean([r.get(key + "_target_score", 0.0)
                                                          for r in sub])) if sub else 0.0, 6)}
        return out_s

    def _agg_base(recs, keys):
        o = {}
        for k in keys:
            if k in ("ngram2", "ngram3", "bm25"):
                o[k] = _bagg(recs, k)
        return o

    base_out = {
        "label": "post-hoc (Lead 评审后追加, 非预注册)",
        "retrieval_by_overlap_layer": _agg_base(base_q, ("ngram2", "ngram3", "bm25")),
        "pair_similarity": {
            "synonym_zero_shared": {
                "n": sum(1 for r in base_pair if r["kind"] == "synonym" and r["zero_shared"]),
                "ngram2_mean": round(float(np.mean([r["ngram2_cos"] for r in base_pair
                                                    if r["kind"] == "synonym" and r["zero_shared"]])), 6),
                "ngram3_mean": round(float(np.mean([r["ngram3_cos"] for r in base_pair
                                                    if r["kind"] == "synonym" and r["zero_shared"]])), 6),
                "bm25_sym_mean": round(float(np.mean([r["bm25_sym"] for r in base_pair
                                                      if r["kind"] == "synonym" and r["zero_shared"]])), 6)},
            "synonym_shared_char": {
                "n": sum(1 for r in base_pair if r["kind"] == "synonym" and not r["zero_shared"]),
                "ngram2_mean": round(float(np.mean([r["ngram2_cos"] for r in base_pair
                                                    if r["kind"] == "synonym" and not r["zero_shared"]])), 6),
                "ngram3_mean": round(float(np.mean([r["ngram3_cos"] for r in base_pair
                                                    if r["kind"] == "synonym" and not r["zero_shared"]])), 6),
                "bm25_sym_mean": round(float(np.mean([r["bm25_sym"] for r in base_pair
                                                      if r["kind"] == "synonym" and not r["zero_shared"]])), 6)},
            "unrelated": {
                "n": sum(1 for r in base_pair if r["kind"] == "unrelated"),
                "ngram2_mean": round(float(np.mean([r["ngram2_cos"] for r in base_pair
                                                    if r["kind"] == "unrelated"])), 6),
                "ngram3_mean": round(float(np.mean([r["ngram3_cos"] for r in base_pair
                                                    if r["kind"] == "unrelated"])), 6),
                "bm25_sym_mean": round(float(np.mean([r["bm25_sym"] for r in base_pair
                                                      if r["kind"] == "unrelated"])), 6)}},
        "query_detail": base_q,
        "pair_detail": base_pair,
        "isolation_anomaly": {"zero_layer_ngram2_hit1": n2_hit_z,
                              "zero_layer_ngram3_hit1": n3_hit_z,
                              "zero_layer_bm25_hit1": bm_hit_z},
        "bm25_pinned": {"k1": 1.5, "b": 0.75, "idf": "ln(1+(N-df+0.5)/(df+0.5))",
                        "corpus": "peer-01-only 8844 cleaned", "doc_len": "bigram tokens incl repeats"},
        "elapsed_s": round(time.perf_counter() - tb0, 1)}
    if n2_hit_z or n3_hit_z or bm_hit_z:
        print("[f3] !!! ISOLATION ANOMALY: baseline hit in zero-overlap layer "
              "(n2=%d n3=%d bm25=%d) - STOP per Lead instruction" % (n2_hit_z, n3_hit_z, bm_hit_z),
              flush=True)
    else:
        print("[f3] baselines done, zero-overlap layer: 0 hits (prediction holds) (%.0fs)"
              % (time.perf_counter() - tb0), flush=True)

    out = {"store_note": F3_STORE_NOTE, "assets": {"synpairs": F3_SYN_PATH,
            "paraphrases": F3_PARA_PATH}, "hc_seeds": F3_HC_SEEDS,
            "first_obs_ne_row": fo_mismatch, "arms": arms_out,
            "comparisons": comparisons, "baselines_posthoc": base_out,
            "completed_at": now_iso()}
    merge_key(RESULT_PATH, "f3", out)
    for nm in F3_ARM_ORDER:
        a = out["arms"][nm]
        aggs = {h: a["seeds"][h]["query_agg"] for h in a["seeds"]}
        print("[f3] arm %s syn_in=%.4f unr_in=%.4f written=%s exact=%s content=%s"
              % (nm, a["syn_input_mean"] or -1, a["unrel_input_mean"] or -1,
                 [a["seeds"][h]["n_written"] for h in a["seeds"]],
                 [aggs[h]["exact_hit"] for h in aggs],
                 [aggs[h]["content_hit"] for h in aggs]), flush=True)
    h2h = out["comparisons"]["zero_shared_synpair_head_to_head"]["summary"]
    print("[f3] zero-shared head-to-head input wins NEW=%d R256=%d | code wins NEW=%d R256=%d"
          % (h2h["NEW_wins_input"], h2h["R256_wins_input"],
             h2h["NEW_wins_code"], h2h["R256_wins_code"]), flush=True)
    qp = out["comparisons"]["query_paired_NEW_vs_R256"]
    print("[f3] paired exact_hit diff (NEW-R256): %s"
          % {h: (qp[h]["exact_hit(1-0)"]["mean_diff"], qp[h]["exact_hit(1-0)"]["p_two_sided"])
             for h in qp}, flush=True)
    print("[f3] STAGE_OK", flush=True)
    return out


# ------------------------------------------------------------- 统计（F3 复用）

def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """单侧 P(T>t)，t 分布自由度 df。t<0 时用对称性：P(T>-|t|)=1-0.5*I_x。"""
    x = df / (df + t * t)
    half = 0.5 * _betainc(df / 2.0, 0.5, x)   # = P(T > |t|)
    return half if t >= 0 else 1.0 - half


def t_cdf(t, df):
    if t >= 0:
        return 1.0 - t_sf(t, df)
    return t_sf(-t, df)


def t_ppf(p, df):
    """CDF 分位数（t_ppf(0.975,10)=2.2281）。二分。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0,1)")
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_t(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    n = int(d.size)
    mean = float(d.mean()) if n else None
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "t": None, "df": None,
                "p_two": None, "ci95": None}
    sd = float(d.std(ddof=1))
    if sd <= 0:
        return {"n": n, "mean": mean, "sd": sd, "t": None, "df": n - 1,
                "p_two": None, "ci95": [mean, mean]}
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    p = 2.0 * t_sf(abs(t), df)
    tc = t_ppf(0.975, df)
    half = tc * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "t": t, "df": df, "p_two": p,
            "ci95": [mean - half, mean + half]}


def bootstrap_ci_mean(vals, n_boot=10000, seed=SEEDS["boot"], alpha=0.05):
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return [float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0))]


# ---------------------------------------------------------------- selftest

def run_selftest():
    res = {"pass": True, "tests": [], "started_at": now_iso()}

    def check(name, cond, detail=""):
        res["tests"].append({"name": name, "pass": bool(cond), "detail": str(detail)})
        if not cond:
            res["pass"] = False

    # ---- judge 纯 mock 单测（L-035 三坑 + 拒写/弃权不误计）
    check("judge_abstain_is_miss", not judge("abstain", 3, {3: 0}, 0))
    check("judge_neg_idx_is_miss", not judge("recall", -1, {0: 0}, 0))
    check("judge_wrong_slot_is_miss", not judge("recall", 7, {7: 5}, 0))
    check("judge_unwritten_target_is_miss", not judge("recall", 2, {2: 9}, 0))
    check("judge_exact_hit", judge("recall", 5, {5: 0}, 0))
    check("judge_separate_is_hit", judge("separate", 5, {5: 0}, 0))
    # 内容等价 vs 精确事件分离（mock payload）
    xt = np.zeros(4, dtype=np.float32); xt[0] = 1.0
    pmat = np.zeros((4, 2), dtype=np.float32)
    pmat[:, 0] = xt            # slot0 = 目标向量（dup 副本等价）
    pmat[1, 1] = 1.0           # slot1 = 其他向量
    check("content_eq_same_vec", content_eq(pmat[:, 0], xt))
    check("content_eq_diff_vec", not content_eq(pmat[:, 1], xt))
    # 场景：目标 obs=0 写在 slot5；同文本 dup 写在 slot9（obs=3）。
    # 检索返回 slot9 -> 精确事件 miss，内容等价 hit。
    obs = {5: 0, 9: 3}
    check("dup_case_exact_miss", not judge("recall", 9, obs, 0))
    check("dup_case_content_hit", content_eq(pmat[:, 0], xt))

    # ---- t 分布 / 配对 t / bootstrap
    check("t_sf_2_10", abs(t_sf(2.0, 10) - 0.0366940) < 2e-5, "got %.8f" % t_sf(2.0, 10))
    check("t_sf_2_30", abs(t_sf(2.0, 30) - 0.0273125225) < 2e-7, "got %.8f" % t_sf(2.0, 30))
    check("t_ppf_975_10", abs(t_ppf(0.975, 10) - 2.2281389) < 1e-4,
          "got %.8f" % t_ppf(0.975, 10))
    check("t_sf_zero_is_half", abs(t_sf(0.0, 7) - 0.5) < 1e-12)
    check("t_sf_symmetry", abs(t_sf(-1.5, 8) - (1.0 - t_sf(1.5, 8))) < 1e-12)
    check("t_sf_monotone", t_sf(1.0, 10) > t_sf(2.0, 10) > t_sf(3.0, 10))
    pt = paired_t([1.0, 2.0, 3.0, 4.0, 5.0])
    exp_t = 3.0 / (math.sqrt(2.5) / math.sqrt(5))
    exp_ci_lo = 3.0 - 2.7764451 * math.sqrt(2.5) / math.sqrt(5)
    exp_ci_hi = 3.0 + 2.7764451 * math.sqrt(2.5) / math.sqrt(5)
    check("paired_t_known_t", abs(pt["t"] - exp_t) < 1e-6, "got %.8f exp %.8f" % (pt["t"], exp_t))
    check("paired_t_known_p", abs(pt["p_two"] - 2.0 * t_sf(exp_t, 4)) < 1e-12)
    check("paired_t_known_ci", abs(pt["ci95"][0] - exp_ci_lo) < 1e-4
          and abs(pt["ci95"][1] - exp_ci_hi) < 1e-4,
          "got %s" % pt["ci95"])
    v = [1.0] * 50 + [5.0] * 50
    b1 = bootstrap_ci_mean(v, n_boot=2000, seed=7)
    b2 = bootstrap_ci_mean(v, n_boot=2000, seed=7)
    check("bootstrap_deterministic", b1 == b2, "%s vs %s" % (b1, b2))
    check("bootstrap_covers_mean", b1[0] < 3.0 < b1[1], "%s" % b1)
    check("bootstrap_reasonable", 2.2 < b1[0] and b1[1] < 3.8, "%s" % b1)

    # ---- 小型真实 HCM 集成（写/拒写/检索）
    try:
        rng = np.random.default_rng(3)
        D = 32
        n = 20
        Xs = np.asarray([unit(rng.standard_normal(D).astype(np.float32)) for _ in range(n)],
                        dtype=np.float32)
        m = HCM(Config(d_x=D, d_c=D, a=0.02, capacity=n + 10, beta=100.0, L=3,
                       use_resid=False, lam=1.0, seed=0))
        for i in range(n):
            m.observe(Xs[i])
        rec_dup = m.observe(Xs[5].copy())
        obs_of = {s: int(m.E.t[s]) for s in range(int(m.E.n))}
        hits = 0
        for i in range(n):
            o = m.query(Xs[i], do_rif=False)
            if judge(o["action"], o.get("idx"), obs_of, i):
                hits += 1
        acts = set()
        for _ in range(5):
            qv = unit(rng.standard_normal(D).astype(np.float32))
            acts.add(str(m.query(qv, do_rif=False)["action"]))
        res["tiny_hcm"] = {"n_written": int(m.E.n), "dup_wrote": bool(rec_dup["wrote"]),
                           "deg0_hits": hits, "novel_actions": sorted(acts)}
        check("tiny_hcm_written_20", int(m.E.n) == 20, "n=%d" % int(m.E.n))
        check("tiny_hcm_dup_rejected", not rec_dup["wrote"])
        check("tiny_hcm_deg0_hits", hits >= 12, "hits=%d/20" % hits)
        check("tiny_hcm_actions_valid", acts <= {"recall", "separate", "abstain"},
              "acts=%s" % sorted(acts))
    except Exception:
        res["tiny_hcm"] = {"error": traceback.format_exc()}
        check("tiny_hcm_no_exception", False)

    res["finished_at"] = now_iso()
    merge_key(RESULT_PATH, "selftest", res)
    npass = sum(1 for t in res["tests"] if t["pass"])
    print("selftest: %d/%d pass, overall=%s" % (npass, len(res["tests"]), res["pass"]),
          flush=True)
    for t in res["tests"]:
        if not t["pass"]:
            print("  FAIL %s %s" % (t["name"], t["detail"]), flush=True)
    if not res["pass"]:
        sys.exit(1)


# -------------------------------------------------------------------- main

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="exp29 F2 text-frontend comparison")
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "smoke", "diag", "full", "f3"])
    ap.add_argument("--arm", default=None, choices=ARM_ORDER,
                    help="full 阶段单臂运行（省内存/断点续跑）")
    ap.add_argument("--job-id", default=None)
    args = ap.parse_args()
    if args.job_id:
        print("job_id=%s" % args.job_id, flush=True)
    t0 = time.perf_counter()
    merge_key(RESULT_PATH, "meta", stage_meta())
    if args.stage == "selftest":
        run_selftest()
    elif args.stage == "smoke":
        run_stage1("smoke", 1000, [0.3])
    elif args.stage == "diag":
        run_stage1("diag", 2000, [0.0, 0.3, 0.5])
    elif args.stage == "full":
        run_full(args.arm)
    elif args.stage == "f3":
        run_f3()
    print("STAGE_OK stage=%s elapsed=%.1fs" % (args.stage, time.perf_counter() - t0),
          flush=True)


if __name__ == "__main__":
    main()
