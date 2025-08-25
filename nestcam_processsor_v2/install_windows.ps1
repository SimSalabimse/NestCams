# NestCam Processor v2.0 - Windows Installation Script

Write-Host "🐦 NestCam Processor v2.0 - Windows Installation" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if Python 3.11 is installed
$python311 = Get-Command python3.11 -ErrorAction SilentlyContinue
if (-not $python311) {
    Write-Host "🐍 Installing Python 3.11..." -ForegroundColor Yellow
    # Download and install Python 3.11
    $pythonUrl = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    $installerPath = "$env:TEMP\python311.exe"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
    Remove-Item $installerPath
} else {
    Write-Host "✅ Python 3.11 already installed" -ForegroundColor Green
}

# Install ffmpeg
Write-Host "🎬 Installing ffmpeg..." -ForegroundColor Yellow
# Download ffmpeg
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$ffmpegZip = "$env:TEMP\ffmpeg.zip"
$ffmpegDir = "$env:TEMP\ffmpeg"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip
Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir
# Add ffmpeg to PATH (you might need to restart your terminal)
$ffmpegPath = Get-ChildItem -Path $ffmpegDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
if ($ffmpegPath) {
    $env:PATH += ";$($ffmpegPath.DirectoryName)"
}

# Create virtual environment
Write-Host "🌐 Creating Python virtual environment..." -ForegroundColor Yellow
python3.11 -m venv nestcam_env

# Activate virtual environment
Write-Host "🔗 Activating virtual environment..." -ForegroundColor Yellow
& ".\nestcam_env\Scripts\activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (if NVIDIA GPU is available)
Write-Host "🔥 Installing PyTorch..." -ForegroundColor Yellow
$nvidia = Get-WmiObject -Query "SELECT * FROM Win32_VideoController WHERE Name LIKE '%NVIDIA%'"
if ($nvidia) {
    Write-Host "🎯 NVIDIA GPU detected, installing CUDA version..." -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "📺 No NVIDIA GPU detected, installing CPU version..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install project dependencies
Write-Host "📚 Installing project dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To run the application:" -ForegroundColor Cyan
Write-Host "   .\nestcam_env\Scripts\activate.ps1" -ForegroundColor White
Write-Host "   cd C:\path\to\nestcam_processsor_v2" -ForegroundColor White
Write-Host "   python -m src.main --web" -ForegroundColor White
