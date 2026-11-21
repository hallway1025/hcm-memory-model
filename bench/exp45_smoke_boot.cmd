@echo off
cd /d "%~dp0.."
py -X utf8 -u bench\exp45_online_beta.py --smoke --out results\exp45_smoke.json > results\exp45_smoke.log 2>&1
echo %ERRORLEVEL% > results\exp45_smoke_exit.txt
