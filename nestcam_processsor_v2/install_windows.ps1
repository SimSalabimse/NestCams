# NestCam Processor v2.0 - Windows Installation Script with Verbose Output

param(
    [switch]$CleanInstall = $false,
    [switch]$Verbose = $true,
    [switch]$SkipTests = $false
)

# Configuration
$Config = @{
    PythonVersion = "3.11.8"
    PythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
    FFmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    TempDir = $env:TEMP
    LogFile = "$PSScriptRoot\install_log.txt"
}

# Start logging
Start-Transcript -Path $Config.LogFile -Append
Write-Host "📝 Installation log: $($Config.LogFile)" -ForegroundColor Cyan

function Write-Step {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Green
    Write-Host ""
}

function Write-SubStep {
    param([string]$Message)
    Write-Host "  → $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "  ✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "  ❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    if ($Verbose) {
        Write-Host "  ℹ️  $Message" -ForegroundColor Blue
    }
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Header
Write-Step "🐦 NestCam Processor v2.0 - Windows Installation"
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

# Get project directory
$PROJECT_DIR = $PSScriptRoot
$VENV_DIR = Join-Path $PROJECT_DIR "nestcam_env"

Write-Info "📂 Project directory: $PROJECT_DIR"
Write-Info "🌐 Virtual environment will be: $VENV_DIR"
Write-Info "📝 Log file: $($Config.LogFile)"

# Check Windows version
$osInfo = Get-ComputerInfo
Write-Info "🖥️  Windows version: $($osInfo.WindowsProductName) $($osInfo.WindowsVersion)"
Write-Info "🔧 PowerShell version: $($PSVersionTable.PSVersion)"

# Check if running as administrator
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Step "⚠️  Administrator privileges required for installation"
    Write-Info "Some installation steps require admin rights (Python, ffmpeg)"
    $adminChoice = Read-Host "Do you want to continue without admin rights? (y/N)"
    if ($adminChoice -notmatch "^[Yy]") {
        Write-Error "Installation cancelled - administrator rights required"
        Stop-Transcript
        exit 1
    }
}

# Step 1: Check Python installation
Write-Step "🐍 Checking Python installation"

$python311 = Test-Command "python3.11"
$python = Test-Command "python"

if ($python311) {
    $pythonVersion = & python3.11 --version
    Write-Success "Python 3.11 found: $pythonVersion"
} elseif ($python) {
    $pythonVersion = & python --version
    Write-Info "Python found: $pythonVersion"
    
    if ($pythonVersion -match "Python 3\.11") {
        Write-Success "Compatible Python version found"
        $python311 = $true
    } else {
        Write-Info "Python version may not be compatible, will install Python 3.11"
    }
} else {
    Write-Info "Python 3.11 not found, will install it"
}

# Install Python if needed
if (-not $python311) {
    Write-Step "📦 Installing Python 3.11"
    
    try {
        $installerPath = Join-Path $Config.TempDir "python311.exe"
        
        Write-SubStep "Downloading Python 3.11..."
        Invoke-WebRequest -Uri $Config.PythonUrl -OutFile $installerPath -UseBasicParsing
        
        Write-SubStep "Installing Python 3.11 (this may take a few minutes)..."
        Write-Info "Installation arguments: /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_venv=1"
        
        $process = Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_venv=1" -Wait -PassThru
        
        if ($process.ExitCode -eq 0) {
            Write-Success "Python 3.11 installed successfully"
        } else {
            Write-Error "Python installation failed with exit code: $($process.ExitCode)"
            throw "Python installation failed"
        }
        
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Clean up
        Remove-Item $installerPath -ErrorAction SilentlyContinue
        
    } catch {
        Write-Error "Failed to install Python 3.11: $_"
        Write-Info "You can download Python manually from: https://www.python.org/downloads/"
        throw
    }
}

# Verify Python installation
Write-SubStep "Verifying Python installation..."
try {
    $pythonVersion = & python3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Python 3.11 verified: $pythonVersion"
    } else {
        throw "Python verification failed"
    }
} catch {
    Write-Error "Python verification failed: $_"
    throw
}

# Step 2: Install ffmpeg
Write-Step "🎬 Installing ffmpeg"

try {
    $ffmpegZip = Join-Path $Config.TempDir "ffmpeg.zip"
    $ffmpegDir = Join-Path $Config.TempDir "ffmpeg"
    
    Write-SubStep "Downloading ffmpeg..."
    Invoke-WebRequest -Uri $Config.FFmpegUrl -OutFile $ffmpegZip -UseBasicParsing
    
    Write-SubStep "Extracting ffmpeg..."
    if (Test-Path $ffmpegDir) {
        Remove-Item $ffmpegDir -Recurse -Force
    }
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir
    
    # Find ffmpeg executable
    $ffmpegPath = Get-ChildItem -Path $ffmpegDir -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ffmpegPath) {
        $ffmpegBinDir = Split-Path $ffmpegPath.FullName -Parent
        Write-Info "ffmpeg found at: $ffmpegBinDir"
        
        # Add to current session PATH
        $env:PATH += ";$ffmpegBinDir"
        
        # Test ffmpeg
        $ffmpegVersion = & ffmpeg -version 2>&1 | Select-Object -First 1
        Write-Success "ffmpeg installed: $ffmpegVersion"
    } else {
        throw "ffmpeg.exe not found after extraction"
    }
    
    # Clean up
    Remove-Item $ffmpegZip -ErrorAction SilentlyContinue
    
} catch {
    Write-Error "Failed to install ffmpeg automatically: $_"
    Write-Info "Please install ffmpeg manually:"
    Write-Info "1. Visit: https://ffmpeg.org/download.html"
    Write-Info "2. Download the Windows build"
    Write-Info "3. Add ffmpeg.exe to your system PATH"
}

# Step 3: Clean up existing virtual environment
if ($CleanInstall -or (Test-Path $VENV_DIR)) {
    Write-Step "🧹 Managing existing virtual environment"
    
    if (Test-Path $VENV_DIR) {
        Write-SubStep "Removing existing virtual environment..."
        try {
            Remove-Item $VENV_DIR -Recurse -Force
            Write-Success "Existing virtual environment removed"
        } catch {
            Write-Error "Failed to remove existing virtual environment: $_"
            throw
        }
    }
}

# Step 4: Create virtual environment
Write-Step "🌐 Creating Python virtual environment"

try {
    Write-SubStep "Creating virtual environment at: $VENV_DIR"
    $venvProcess = Start-Process -FilePath "python3.11" -ArgumentList "-m", "venv", $VENV_DIR -Wait -PassThru -NoNewWindow
    
    if ($venvProcess.ExitCode -eq 0) {
        Write-Success "Virtual environment created successfully"
    } else {
        throw "Virtual environment creation failed with exit code: $($venvProcess.ExitCode)"
    }
    
} catch {
    Write-Error "Failed to create virtual environment: $_"
    throw
}

# Verify venv creation
$activateScript = Join-Path $VENV_DIR "Scripts\activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment creation failed - activate script not found at: $activateScript"
    throw "Virtual environment creation failed"
}

# Step 5: Activate virtual environment
Write-Step "🔗 Activating virtual environment"

try {
    Write-SubStep "Running activation script..."
    & $activateScript
    
    # Verify activation
    if ($env:VIRTUAL_ENV) {
        Write-Success "Virtual environment activated: $env:VIRTUAL_ENV"
    } else {
        throw "Virtual environment activation failed - VIRTUAL_ENV not set"
    }
    
} catch {
    Write-Error "Failed to activate virtual environment: $_"
    throw
}

# Step 6: Upgrade pip
Write-Step "⬆️ Upgrading pip"

try {
    Write-SubStep "Upgrading pip in virtual environment..."
    $pipUpgrade = Start-Process -FilePath "python" -ArgumentList "-m", "pip", "install", "--upgrade", "pip" -Wait -PassThru -NoNewWindow
    
    if ($pipUpgrade.ExitCode -eq 0) {
        $pipVersion = & pip --version
        Write-Success "pip upgraded successfully: $pipVersion"
    } else {
        Write-Error "pip upgrade failed with exit code: $($pipUpgrade.ExitCode)"
        throw "pip upgrade failed"
    }
    
} catch {
    Write-Error "Failed to upgrade pip: $_"
    throw
}

# Step 7: Install PyTorch
Write-Step "🔥 Installing PyTorch"

try {
    # Check for NVIDIA GPU
    Write-SubStep "Checking for NVIDIA GPU..."
    $nvidia = Get-WmiObject -Query "SELECT * FROM Win32_VideoController WHERE Name LIKE '%NVIDIA%'" -ErrorAction SilentlyContinue
    
    if ($nvidia) {
        Write-Info "🎯 NVIDIA GPU detected: $($nvidia.Name)"
        Write-SubStep "Installing PyTorch with CUDA support..."
        $torchArgs = "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
    } else {
        Write-Info "📺 No NVIDIA GPU detected, installing CPU version"
        Write-SubStep "Installing PyTorch CPU version..."
        $torchArgs = "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
    }
    
    $torchInstall = Start-Process -FilePath "pip" -ArgumentList $torchArgs -Wait -PassThru -NoNewWindow
    
    if ($torchInstall.ExitCode -eq 0) {
        Write-Success "PyTorch installed successfully"
    } else {
        Write-Error "PyTorch installation failed with exit code: $($torchInstall.ExitCode)"
        throw "PyTorch installation failed"
    }
    
} catch {
    Write-Error "Failed to install PyTorch: $_"
    throw
}

# Step 8: Install project dependencies
Write-Step "📚 Installing project dependencies"

try {
    Write-SubStep "Installing dependencies from requirements.txt..."
    $depsInstall = Start-Process -FilePath "pip" -ArgumentList "install", "-r", "$PROJECT_DIR\requirements.txt" -Wait -PassThru -NoNewWindow
    
    if ($depsInstall.ExitCode -eq 0) {
        Write-Success "Project dependencies installed successfully"
    } else {
        Write-Error "Dependencies installation failed with exit code: $($depsInstall.ExitCode)"
        throw "Dependencies installation failed"
    }
    
} catch {
    Write-Error "Failed to install project dependencies: $_"
    throw
}

# Step 9: Test installation
if (-not $SkipTests) {
    Write-Step "🧪 Testing installation"
    
    try {
        Write-SubStep "Testing Python imports..."
        
        $testScript = @"
import sys
print(f'Python version: {sys.version}')

# Test core dependencies
deps_to_test = [
    ('torch', 'PyTorch'),
    ('cv2', 'OpenCV'),
    ('streamlit', 'Streamlit'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('plotly', 'Plotly'),
    ('psutil', 'psutil'),
]

failed_deps = []
for module, name in deps_to_test:
    try:
        __import__(module)
        print(f'✅ {name} available')
    except ImportError as e:
        print(f'❌ {name} failed: {e}')
        failed_deps.append((module, name))

if failed_deps:
    print(f'\n❌ {len(failed_deps)} dependencies failed to import')
    for module, name in failed_deps:
        print(f'  - {name} ({module})')
    sys.exit(1)
else:
    print('\n✅ All dependencies imported successfully')
"@

        $testResult = $testScript | & python
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Installation test completed successfully"
            Write-Info "All core dependencies are working"
        } else {
            Write-Error "Installation test failed"
            Write-Info "Check the log file for detailed error information"
            throw "Installation test failed"
        }
        
    } catch {
        Write-Error "Installation test failed: $_"
        throw
    }
} else {
    Write-Info "Skipping installation tests (-SkipTests parameter used)"
}

# Step 10: Create activation helper script
Write-Step "📝 Creating helper scripts"

try {
    $activateHelper = @'
# NestCam Processor v2.0 - Virtual Environment Activator

param(
    [switch]$RunApp = $false
)

$PROJECT_DIR = Split-Path -Parent $PSCommandPath
$VENV_DIR = Join-Path $PROJECT_DIR "nestcam_env"

Write-Host "🔗 Activating NestCam virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path $VENV_DIR)) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run the installation script first:" -ForegroundColor Yellow
    Write-Host "   .\install_windows.ps1" -ForegroundColor White
    exit 1
}

& "$VENV_DIR\Scripts\activate.ps1"

if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
    Write-Host ""
    
    if ($RunApp) {
        Write-Host "🚀 Starting NestCam Processor..." -ForegroundColor Cyan
        python -m src.main --web
    } else {
        Write-Host "🚀 To run the application:" -ForegroundColor Cyan
        Write-Host "   python -m src.main --web" -ForegroundColor White
        Write-Host ""
        Write-Host "📝 To deactivate later:" -ForegroundColor Cyan
        Write-Host "   deactivate" -ForegroundColor White
    }
} else {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
'@

    $activateHelper | Out-File -FilePath "$PROJECT_DIR\activate_venv.ps1" -Encoding UTF8
    Write-Success "Helper script created: activate_venv.ps1"
    
} catch {
    Write-Error "Failed to create helper script: $_"
}

# Final summary
Write-Step "🎉 Installation Complete!"

Write-Host ""
Write-Host "📋 Installation Summary:" -ForegroundColor Cyan
Write-Host "  📂 Project directory: $PROJECT_DIR" -ForegroundColor White
Write-Host "  🌐 Virtual environment: $VENV_DIR" -ForegroundColor White
Write-Host "  📝 Log file: $($Config.LogFile)" -ForegroundColor White
Write-Host "  🛠️  Helper script: .\activate_venv.ps1" -ForegroundColor White

Write-Host ""
Write-Host "🚀 How to run NestCam Processor:" -ForegroundColor Cyan
Write-Host "  Option 1 - Use helper script:" -ForegroundColor White
Write-Host "    .\activate_venv.ps1 -RunApp" -ForegroundColor Gray
Write-Host ""
Write-Host "  Option 2 - Manual activation:" -ForegroundColor White
Write-Host "    .\activate_venv.ps1" -ForegroundColor Gray
Write-Host "    python -m src.main --web" -ForegroundColor Gray

Write-Host ""
Write-Host "📚 For detailed documentation, see: README.md" -ForegroundColor Cyan
Write-Host "🆘 For help, check the troubleshooting section in README.md" -ForegroundColor Cyan

Stop-Transcript
