$ErrorActionPreference = 'SilentlyContinue'
Get-Process py, python | ForEach-Object { 'PROC {0} cpu={1}s ws={2}MB' -f $_.Id, [int]($_.CPU), [int]($_.WorkingSet64/1MB) }
if (Test-Path C:/Users/you/hcm/results/exp29_full_m2.log) {
  '--- exp29 log tail ---'
  Get-Content C:/Users/you/hcm/results/exp29_full_m2.log -Tail 10
} else { 'no exp29 log' }
$rj = 'C:/Users/you/hcm/results/exp29_text_frontend_v2.json'
if (Test-Path $rj) { 'RESULT_JSON size=' + (Get-Item $rj).Length } else { 'RESULT_JSON not yet' }
if (Test-Path C:/Users/you/hcm/results/exp29_boot_error.txt) { '--- boot error ---'; Get-Content C:/Users/you/hcm/results/exp29_boot_error.txt }
