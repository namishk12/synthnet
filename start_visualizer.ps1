$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8765
$maxPort = 8775
$serverReady = $false

while ($port -le $maxPort) {
    $candidateUrl = "http://127.0.0.1:$port/visualizer.html"
    $existing = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $existing) {
        break
    }
    try {
        $response = Invoke-WebRequest -Uri $candidateUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200 -and $response.Content -match "Telecom CGAN") {
            $serverReady = $true
            break
        }
    } catch {
        # The port belongs to another service; try the next one.
    }
    $port += 1
}

if ($port -gt $maxPort) {
    throw "No free local port was found between 8765 and 8775."
}

$url = "http://127.0.0.1:$port/visualizer.html"

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
        -ArgumentList "-m", "http.server", "$port", "--bind", "127.0.0.1" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden

    for ($attempt = 0; $attempt -lt 20; $attempt += 1) {
        Start-Sleep -Milliseconds 250
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $serverReady = $true
                break
            }
        } catch {
            # The server may still be starting.
        }
    }
    if (-not $serverReady) {
        throw "The local visualizer server did not start successfully at $url"
    }
}

Write-Host "Telecom CGAN visualizer: $url"
Start-Process $url
