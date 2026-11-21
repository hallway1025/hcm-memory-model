@echo off
cd /d "%~dp0.."
py -X utf8 -u bench\exp44_online_tag.py --smoke --out results\exp44_smoke.json > results\exp44_smoke.log 2>&1
echo %ERRORLEVEL% > results\exp44_smoke_exit.txt
