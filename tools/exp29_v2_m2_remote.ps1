try {
  Set-Location C:/Users/you/hcm
  $env:OMP_NUM_THREADS = '10'
  $env:OPENBLAS_NUM_THREADS = '10'
  $py = 'C:/Users/you/AppData/Local/Programs/Python/Launcher/py.exe'
  & $py -X utf8 -u bench/exp29_v2_m2.py --stage full --job-id full5m2v2 2>&1 | Out-File -FilePath results/exp29_full_m2_v2.log -Encoding utf8
  $LASTEXITCODE | Set-Content -Encoding ASCII results/exp29_v2_exit.txt
} catch {
  $_ | Out-String | Out-File results/exp29_v2_boot_error.txt -Encoding utf8
}
