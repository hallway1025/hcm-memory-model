
Set-Location C:/Users/you/hcm
if (Test-Path results/exp29_text_frontend.json) { Copy-Item results/exp29_text_frontend.json results/exp29_text_frontend_v1.json -Force; Write-Output 'v1 JSON backed up' } else { Write-Output 'no v1 JSON' }
py -X utf8 -c "import sys; sys.path.insert(0,'bench'); import exp29_text_frontend as e; print('TRAIN', len(e.TRAIN_FILES), e.TRAIN_FILES[0], '|', e.TRAIN_FILES[-1]); print('OUT default', e.RESULT_PATH)"
Write-Output ("HASH " + (Get-FileHash bench/exp29_text_frontend.py -Algorithm SHA256).Hash)
Write-Output ("NPZ-HASH " + (Get-FileHash bench/exp23_semantic_embed.npz -Algorithm SHA256).Hash)
