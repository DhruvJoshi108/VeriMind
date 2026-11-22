$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$packageTarget = Join-Path (Split-Path $projectRoot -Parent) "gpu-torch-packages"
$sitePackages = Join-Path $projectRoot "venv\Lib\site-packages"
$envPath = Join-Path $projectRoot ".env"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project venv not found. Create it with: py -3.11 -m venv venv"
}
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw "NVIDIA driver tools not found; install/update the NVIDIA driver first."
}

$null = New-Item -ItemType Directory -Path $packageTarget -Force
$torchStatus = @"
import sys
sys.path.insert(0, r'$packageTarget')
try:
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
except Exception:
    print("unavailable")
"@ | & $python -
$hasCudaTorch = $LASTEXITCODE -eq 0 -and $torchStatus.Count -ge 2 `
    -and $torchStatus[0] -eq "2.14.1+cu130" -and $torchStatus[1] -eq "True"
if (-not $hasCudaTorch) {
    & $python -m pip install --ignore-installed --no-deps --target $packageTarget `
        "torch==2.14.1+cu130" `
        --index-url "https://download.pytorch.org/whl/cu130" `
        --extra-index-url "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed."
    }
}

$sentencePieceStatus = @"
import sys
sys.path.insert(0, r'$packageTarget')
try:
    import sentencepiece
    print("present")
except ImportError:
    print("missing")
"@ | & $python -
if ($LASTEXITCODE -ne 0 -or $sentencePieceStatus -notcontains "present") {
    & $python -m pip install --ignore-installed --no-deps --target $packageTarget `
        "sentencepiece>=0.2.0" `
        --index-url "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) {
        throw "SentencePiece installation failed."
    }
}

$protobufStatus = @"
import sys
sys.path.insert(0, r'$packageTarget')
try:
    import google.protobuf
    print("present")
except ImportError:
    print("missing")
"@ | & $python -
if ($LASTEXITCODE -ne 0 -or $protobufStatus -notcontains "present") {
    & $python -m pip install --ignore-installed --no-deps --target $packageTarget `
        "protobuf>=4.25" `
        --index-url "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) {
        throw "Protobuf installation failed."
    }
}

$pthPath = Join-Path $sitePackages "verimind_gpu_packages.pth"
Set-Content -LiteralPath $pthPath `
    -Value "import sys; sys.path.insert(0, r'$packageTarget')" `
    -Encoding ascii

$envLines = New-Object 'System.Collections.Generic.List[string]'
if (Test-Path -LiteralPath $envPath) {
    foreach ($line in Get-Content -LiteralPath $envPath) {
        [void]$envLines.Add($line)
    }
}
$nliSettings = [ordered]@{
    NLI_BACKEND = "transformer"
    NLI_DEVICE = "cuda"
    NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli"
}
foreach ($key in $nliSettings.Keys) {
    $pattern = "^\s*$([regex]::Escape($key))\s*="
    $found = $false
    for ($index = 0; $index -lt $envLines.Count; $index++) {
        if ($envLines[$index] -match $pattern) {
            $envLines[$index] = "$key=$($nliSettings[$key])"
            $found = $true
            break
        }
    }
    if (-not $found) {
        [void]$envLines.Add("$key=$($nliSettings[$key])")
    }
}
[System.IO.File]::WriteAllLines($envPath, $envLines, [System.Text.UTF8Encoding]::new($false))

& $python -c "import torch; assert torch.cuda.is_available(); print(f'GPU ready: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__}, CUDA {torch.version.cuda})')"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installed, but CUDA verification failed."
}

Write-Output "Transformer NLI enabled in .env. Model weights will download on first API start."
