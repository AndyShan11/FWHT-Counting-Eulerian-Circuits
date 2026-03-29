# ===========================================================================
# RQ3: Effect of Graph Size at Fixed Circuit Rank
# Generates graphs, compiles the FWHT solver, runs experiments, plots results
# ===========================================================================

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$SRC_FILE  = "$RepoRoot\src\fwht_solver.cpp"
$EXEC_FILE = "$RepoRoot\fwht_solver.exe"
$GRAPH_DIR = "$RepoRoot\data\rq3_graphs"
$CSV_OUT   = "$RepoRoot\results\eulerian_RQ3_results.csv"

# Ensure output directories exist
New-Item -ItemType Directory -Path "$RepoRoot\results" -Force | Out-Null
New-Item -ItemType Directory -Path "$RepoRoot\figures"  -Force | Out-Null

# --- Step 1: Generate test graphs ---
Write-Host "[*] Step 1: Generating RQ3 test graphs ..." -ForegroundColor Cyan
python "$RepoRoot\generators\gen_rq3.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Graph generation failed!" -ForegroundColor Red
    exit
}
Write-Host "[+] Graph generation complete." -ForegroundColor Green

# --- Step 2: Compile ---
Write-Host "`n[*] Step 2: Compiling fwht_solver.cpp ..." -ForegroundColor Cyan
g++ -O3 -funroll-loops -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -fopenmp $SRC_FILE -o $EXEC_FILE
if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Build failed." -ForegroundColor Red
    exit
}
Write-Host "[+] Build successful!" -ForegroundColor Green

# --- Step 3: Run solver on all generated graphs ---
Write-Host "`n[*] Step 3: Running FWHT solver on $GRAPH_DIR ..." -ForegroundColor Cyan
Write-Host "    Output CSV: $CSV_OUT"
Write-Host "    Timeout: 600s per graph"
Write-Host "------------------------------------------------"

& $EXEC_FILE $GRAPH_DIR $CSV_OUT

Write-Host "------------------------------------------------"
Write-Host "[+] RQ3 experiments complete!" -ForegroundColor Green
Write-Host "[+] Results in: $CSV_OUT" -ForegroundColor Green

# --- Step 4: Generate plots ---
Write-Host "`n[*] Step 4: Generating plots ..." -ForegroundColor Cyan
python "$RepoRoot\plotting\plot_rq3.py"
Write-Host "[+] Done!" -ForegroundColor Green
