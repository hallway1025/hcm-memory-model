# HCM v1.4 一键复现环境

**目标**（主人批准件 3 的底线）：外部实验室拿到仓库后，**一条命令**重建环境、跑完全部实验与守卫，得到与本机一致的全绿 verdict。

## 入口（三选一）

**A. Windows 原生（本机已验证路径）**
```powershell
powershell -ExecutionPolicy Bypass -File repro\setup.ps1     # 建 .venv-repro + 钉版依赖
powershell -ExecutionPolicy Bypass -File repro\run_repro.ps1 # 一键：25 步全流程 + 三守卫 + 哈希对账
```

**B. Linux / macOS 原生**
```bash
bash repro/setup.sh        # 建 .venv-repro + 钉版依赖
bash repro/run_repro.sh    # 一键：25 步全流程 + 三守卫 + 哈希对账
```

**C. 容器（Docker）**
```bash
docker build -t hcm-v14 repro/
docker run --rm hcm-v14
```
容器路径本机无 docker、未实测，外部实验室首次构建时自证。

## 验收口径（run_guards.py 自动判定，非零退出即 FAIL）

| 守卫 | 命令 | 判据 |
|---|---|---|
| G1 claims | tools/check_claims.py | 457/457 PASS（三向比对全绿） |
| G2 citations | tools/verify_citations.py verify | n=448 gate PASS（T1+T2=79.5%） |
| G3 closure | tools/update_cited_ids.py | entries=57 mapped=57 unmatched=[] |
| G4 sha256 | audit/SHA256SUMS.txt 对账 | 全部条目哈希一致、无缺失 |

## 依赖（repro/requirements.txt，版本钉死）

numpy==2.4.4 · matplotlib==3.11.2 · python-docx==1.2.0 · pypdf==7.0.0 · requests==2.33.1（Python 3.11+，实测 3.14.0）。

## 网络与镜像

默认走 pypi.org。境内网络（pypi.org 不可达）时先设镜像再跑 setup：

```powershell
$env:PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'   # 或 mirrors.aliyun.com/pypi/simple
powershell -ExecutionPolicy Bypass -File repro\setup.ps1
```

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
bash repro/setup.sh
```

## 关键纪律

- **不要单独跑某个 bench 脚本**：会覆写 results/*.json 使 check_claims 期望失配。要复跑单个实验，之后必须重跑完整 reproduce_all.py 恢复全绿（README 已声明）。
- reproduce_all.py 内含 exp36b（约 25 分钟），全流程约 30–40 分钟，属预期耗时。
- 引文核验（G2）离线使用 refs/cache 缓存，无网络也可跑。
- G4 对账基于 audit/SHA256SUMS.txt（tools/make_manifest.py 生成，清单不含自身）。
