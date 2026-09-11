$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8780
$maxPort = 8790
$serverReady = $false

while ($port -le $maxPort) {
    $healthUrl = "http://127.0.0.1:$port/api/health"
    $existing = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $existing) { break }
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200 -and $response.Content -match '"ok": true') {
            $serverReady = $true
            break
        }
    } catch {
        # This port belongs to another service; try the next one.
    }
    $port += 1
}

if ($port -gt $maxPort) {
    throw "No free local port was found between 8780 and 8790."
}

$url = "http://127.0.0.1:$port/"
if (-not $serverReady) {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $python = $venvPython
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $python = "py"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $python = "python"
    } else {
        throw "Python was not found. Install Python or create the project .venv first."
    }

    Start-Process `
        -FilePath $python `
        -ArgumentList "cgan_gui_server.py", "--port", "$port" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden

    for ($attempt = 0; $attempt -lt 30; $attempt += 1) {
        Start-Sleep -Milliseconds 250
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $serverReady = $true
                break
            }
        } catch {
            # The local server may still be starting.
        }
    }
    if (-not $serverReady) {
        throw "The Telecom CGAN Studio did not start successfully at $url"
    }
}

Write-Host "Telecom CGAN Studio: $url"
Start-Process $url
