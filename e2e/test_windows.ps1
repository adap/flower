python -m pip install -U pip==23.3.1
python -m pip install --upgrade ..

# Create and install Flower app
flwr new e2e-tmp-test --framework numpy --username flwrlabs
Set-Location e2e-tmp-test

# Modify the config file
Add-Content -Path pyproject.toml -Value ""
Add-Content -Path pyproject.toml -Value "[tool.flwr.federations.e2e]"
Add-Content -Path pyproject.toml -Value "address = '127.0.0.1:9093'"
Add-Content -Path pyproject.toml -Value "insecure = true"

# Start Flower processes in the background
Start-Process -NoNewWindow -FilePath flower-superlink -ArgumentList "--insecure" -RedirectStandardOutput flwr_output.log
Start-Sleep -Seconds 2

$cl1 = Start-Process -NoNewWindow -PassThru -FilePath flower-supernode -ArgumentList "--insecure --superlink 127.0.0.1:9092 --clientappio-api-address localhost:9094 --node-config 'partition-id=0 num-partitions=2' --max-retries 0"
Start-Sleep -Seconds 2

Start-Process -NoNewWindow -FilePath flower-supernode -ArgumentList "--insecure --superlink 127.0.0.1:9092 --clientappio-api-address localhost:9095 --node-config 'partition-id=1 num-partitions=2' --max-retries 0"
Start-Sleep -Seconds 2

# Run the Flower command
flwr run --run-config num-server-rounds=1 . e2e

# Function to clean up processes on exit
function Cleanup {
    Write-Output "Stopping Flower processes..."
    Get-Process | Where-Object { $_.ProcessName -like "flower*" } | ForEach-Object { Stop-Process -Id $_.Id -Force }
}

# Register cleanup on exit
$cleanupScript = { Cleanup }
Register-EngineEvent PowerShell.Exiting -Action $cleanupScript

# Initialize a flag to track if training is successful
$found_success = $false
$timeout = 120  # Timeout after 120 seconds
$elapsed = 0

# Check for "Run finished" in a loop with a timeout
while (-not $found_success -and $elapsed -lt $timeout) {
    if (Select-String -Path flwr_output.log -Pattern "Run finished" -Quiet) {
        Write-Output "Training worked correctly!"
        $found_success = $true
        exit 0
    } else {
        Write-Output "Waiting for training ... ($elapsed seconds elapsed)"
    }
    # Sleep for a short period and increment the elapsed time
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $found_success) {
    Write-Output "Training did not finish within timeout."
    exit 1
}
