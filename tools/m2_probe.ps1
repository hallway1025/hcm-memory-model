(Get-FileHash "C:\Users\you\hcm\bench\exp29_text_frontend.py" -Algorithm SHA256).Hash
(Get-FileHash "C:\Users\you\hcm\src\hcm\text_encoder.py" -Algorithm SHA256).Hash
(Get-FileHash "C:\Users\you\hcm\bench\frozen_synpairs.json" -Algorithm SHA256).Hash
(Get-FileHash "C:\Users\you\hcm\bench\frozen_paraphrases.json" -Algorithm SHA256).Hash
(Get-ChildItem "C:\Users\you\hcm\bench\corpus").Count
