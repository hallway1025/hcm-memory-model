Write-Output ('TaskInfo: ' + (Get-ScheduledTaskInfo -TaskName hcm_exp29_v2 | Out-String))
Write-Output ('Action: ' + ((Get-ScheduledTask -TaskName hcm_exp29_v2).Actions | Out-String))
Write-Output ('Principal: ' + ((Get-ScheduledTask -TaskName hcm_exp29_v2).Principal | Out-String))
foreach ($f in @('results/exp29_full_m2_v2.log','results/exp29_v2_exit.txt','results/exp29_v2_boot_error.txt')) {
  $p = 'C:/Users/you/hcm/' + $f; if (Test-Path $p) { Write-Output ($f + ' EXISTS ' + (Get-Item $p).Length + 'B') } else { Write-Output ($f + ' missing') }
}
