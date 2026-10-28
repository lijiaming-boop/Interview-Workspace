param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [int]$Episodes = 50,
    [int]$NumEnvs = 8,
    [string]$IsaacLabLauncher = "isaaclab.bat",
    [string]$OpenVlaRoot = $env:OPENVLA_OFT_ROOT,
    [string]$UnnormKey = "bridge_orig",
    [int]$Seed = 7,
    [string]$OutRoot = "12-IsaacLab-Bridge-Franka/outputs/domain-gap"
)

if (-not $OpenVlaRoot) {
    throw "Set OPENVLA_OFT_ROOT or pass -OpenVlaRoot before running."
}

$common = @(
    "-p", "12-IsaacLab-Bridge-Franka/run_closed_loop.py",
    "--policy", "openvla",
    "--checkpoint", $Checkpoint,
    "--openvla_root", $OpenVlaRoot,
    "--unnorm_key", $UnnormKey,
    "--episodes", "$Episodes",
    "--num_envs", "$NumEnvs",
    "--open_loop_steps", "1",
    "--seed", "$Seed",
    "--headless"
)

& $IsaacLabLauncher @common "--domain_mode" "raw" "--output_dir" "$OutRoot/raw"
if ($LASTEXITCODE -ne 0) { throw "raw evaluation failed with exit code $LASTEXITCODE" }

& $IsaacLabLauncher @common "--domain_mode" "bridge_aligned" "--output_dir" "$OutRoot/bridge_aligned"
if ($LASTEXITCODE -ne 0) { throw "bridge_aligned evaluation failed with exit code $LASTEXITCODE" }

$rawSummary = Get-ChildItem -LiteralPath "$OutRoot/raw" -Filter summary.json -Recurse | Sort-Object LastWriteTime | Select-Object -Last 1
$alignedSummary = Get-ChildItem -LiteralPath "$OutRoot/bridge_aligned" -Filter summary.json -Recurse | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $rawSummary -or -not $alignedSummary) { throw "summary.json not found" }

python "12-IsaacLab-Bridge-Franka/analyze_results.py" `
    --before $rawSummary.FullName `
    --after $alignedSummary.FullName `
    --report "$OutRoot/domain_gap_report.md"
