# HCM v1.2 一键复现容器（外部实验室：docker build -t hcm-v13 repro/ && docker run --rm hcm-v13）
# 本机未装 docker，此路径未在本机实测；构建与验收在外部实验室首次执行时自证。
FROM python:3.14-slim
WORKDIR /hcm
COPY repro/requirements.txt /hcm/repro/requirements.txt
RUN pip install --no-cache-dir -r repro/requirements.txt
COPY . /hcm
# 单入口：25 步全流程 + 验收器（G1-G4）
CMD ["sh", "-c", "python -u bench/reproduce_all.py && python -u repro/run_guards.py"]
