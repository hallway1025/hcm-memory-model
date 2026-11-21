$ErrorActionPreference = 'SilentlyContinue'
Get-Process py, python | ForEach-Object { 'PROC {0} {1}MB' -f $_.Id, [int]($_.WorkingSet64/1MB) }
if (Test-Path C:/Users/you/hcm/results/exp29_full_m2.log) {
  '--- log tail ---'
  Get-Content C:/Users/you/hcm/results/exp29_full_m2.log -Tail 6
} else { 'no exp29 log' }
