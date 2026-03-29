# ===========================================================================
# RQ4: Parallel Scaling -- Multi-threading Speedup
# Tests multiple graph sizes x multiple thread counts, 3 repeats each
# ===========================================================================

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$SRC_FILE  = "$RepoRoot\src\fwht_solver.cpp"
$EXEC_FILE = "$RepoRoot\fwht_solver_rq4.exe"
$GRAPH_DIR = "$RepoRoot\data\rq4_graphs"
$CSV_FILE  = "$RepoRoot\results\rq4_results.csv"
$TEMP_DIR  = "$RepoRoot\data\rq4_temp_single"
$REPEATS   = 3

# Ensure output directories exist
New-Item -ItemType Directory -Path "$RepoRoot\results" -Force | Out-Null
New-Item -ItemType Directory -Path "$RepoRoot\figures"  -Force | Out-Null

# --- Step 0: Generate benchmark graphs ---
Write-Host "[*] Step 0: Generating RQ4 benchmark graphs ..." -ForegroundColor Cyan
python "$RepoRoot\generators\gen_rq4.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Graph generation failed!" -ForegroundColor Red
    exit
}
Write-Host "[+] Graph generation complete.`n" -ForegroundColor Green

# --- Step 1: Compile ---
Write-Host "[*] Step 1: Compiling fwht_solver.cpp ..." -ForegroundColor Cyan
g++ -O3 -funroll-loops -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -fopenmp $SRC_FILE -o $EXEC_FILE
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Build failed." -ForegroundColor Red
    exit
}
Write-Host "[+] Build successful!`n" -ForegroundColor Green

# --- Step 2: Run experiments ---
"Graph,Threads,Run,Time(s)" | Out-File -FilePath $CSV_FILE -Encoding ascii

$threads_list = @(1, 2, 4, 8, 16, 32)

$graphs = Get-ChildItem -Path $GRAPH_DIR -Filter "*.gr" | Sort-Object Name

foreach ($graph in $graphs) {
    $graphName = $graph.Name
    $graphPath = $graph.FullName

    Write-Host "================================================" -ForegroundColor Yellow
    Write-Host ">>> Graph: $graphName" -ForegroundColor Yellow
    Write-Host "================================================"

    foreach ($t in $threads_list) {
        Write-Host "  Threads=$t : " -NoNewline

        $env:OMP_NUM_THREADS = $t
        $allTimes = @()

        for ($r = 1; $r -le $REPEATS; $r++) {
            # Prepare temp directory with single graph file
            if (Test-Path $TEMP_DIR) { Remove-Item -Recurse -Force $TEMP_DIR }
            New-Item -ItemType Directory -Path $TEMP_DIR -Force | Out-Null
            Copy-Item $graphPath -Destination "$TEMP_DIR\$graphName"

            # Run solver, capture stdout
            $output = & $EXEC_FILE $TEMP_DIR 2>&1

            # Parse "Time     : X.XXX s" from stdout
            $timeLine = $output | Where-Object { $_ -match "Time\s+:" }
            if ($timeLine) {
                if ($timeLine -is [array]) { $timeLine = $timeLine[-1] }
                $timeVal = [regex]::Match($timeLine, '([\d.]+)\s*s').Groups[1].Value
                if ($timeVal) {
                    $allTimes += [double]$timeVal
                    "$graphName,$t,$r,$timeVal" | Out-File -FilePath $CSV_FILE -Append -Encoding ascii
                } else {
                    "$graphName,$t,$r,ERROR" | Out-File -FilePath $CSV_FILE -Append -Encoding ascii
                }
            } else {
                Write-Host "X " -NoNewline -ForegroundColor Red
                "$graphName,$t,$r,ERROR" | Out-File -FilePath $CSV_FILE -Append -Encoding ascii
            }
        }

        if ($allTimes.Count -gt 0) {
            $avg = ($allTimes | Measure-Object -Average).Average
            Write-Host ("{0:F3}s (avg over {1} runs)" -f $avg, $allTimes.Count) -ForegroundColor Green
        } else {
            Write-Host "All runs failed" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# Clean up temp directory
if (Test-Path $TEMP_DIR) { Remove-Item -Recurse -Force $TEMP_DIR }

Write-Host "================================================" -ForegroundColor Green
Write-Host "[+] All RQ4 experiments complete!" -ForegroundColor Green
Write-Host "[+] Results saved to $CSV_FILE" -ForegroundColor Green

# --- Step 3: Plot ---
Write-Host "`n[*] Step 3: Generating plots ..." -ForegroundColor Cyan
python "$RepoRoot\plotting\plot_rq4.py"
Write-Host "[+] Done!" -ForegroundColor Green
