$ErrorActionPreference = 'Continue'
Set-Location C:\Users\you\hcm
py -X utf8 bench\exp29_text_frontend.py full --job-id full4m2 *> results\exp29_full_m2.log
"EXITCODE=$LASTEXITCODE" | Out-File results\exp29_m2_exit.txt -Encoding ascii
