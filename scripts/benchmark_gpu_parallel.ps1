param(
	[string]$LogDir = "./benchmark-logs",
	[int[]]$Workers = @(1, 2, 3, 4),
	[int]$Genomes = 20,
	[string]$DatasetProfileId = "",
	[switch]$UseParallelSafeLimited
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $LogDir)) {
	New-Item -ItemType Directory -Path $LogDir | Out-Null
}

Write-Host "GPU baseline benchmark harness"
Write-Host "Genomes: $Genomes"
Write-Host "Workers: $($Workers -join ', ')"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$summaryPath = Join-Path $LogDir "summary-$stamp.csv"
"workers,mode,elapsed_sec,samples_per_sec,speedup_vs_k1,log_file" | Out-File -FilePath $summaryPath -Encoding utf8

$k1Elapsed = $null

foreach ($k in $Workers) {
	$mode = if ($UseParallelSafeLimited) { "parallel-safe-limited" } else { "sequential" }
	if ($k -gt 1) { $mode = "parallel-safe-limited" }

	$runLog = Join-Path $LogDir "run-k$k-$stamp.log"
	Write-Host "Running K=$k mode=$mode ..."

	$sw = [System.Diagnostics.Stopwatch]::StartNew()

	# This harness expects backend logs from `evaluate_population` while app runs under `npm run tauri dev`.
	# It captures stdout/stderr from an external command you provide in BENCHMARK_COMMAND.
	$benchmarkCommand = $env:BENCHMARK_COMMAND
	if ([string]::IsNullOrWhiteSpace($benchmarkCommand)) {
		Write-Host "BENCHMARK_COMMAND is not set. Example: set BENCHMARK_COMMAND=\"npm run tauri dev\""
		Write-Host "Skipping execution and writing placeholder row."
		$sw.Stop()
		"$k,$mode,0,0,0,$runLog" | Out-File -FilePath $summaryPath -Append -Encoding utf8
		continue
	}

	$cmd = "$benchmarkCommand"
	& cmd.exe /c $cmd *>&1 | Tee-Object -FilePath $runLog | Out-Null
	$sw.Stop()

	$elapsedSec = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
	$samplesPerSec = 0.0

	if (Test-Path $runLog) {
		$samplesMatches = Select-String -Path $runLog -Pattern "samples_per_sec[:=]\s*([0-9]+(\.[0-9]+)?)" -AllMatches
		if ($samplesMatches) {
			$vals = @()
			foreach ($m in $samplesMatches.Matches) {
				$vals += [double]$m.Groups[1].Value
			}
			if ($vals.Count -gt 0) {
				$samplesPerSec = [Math]::Round((($vals | Measure-Object -Average).Average), 3)
			}
		}
	}

	if ($k -eq 1) {
		$k1Elapsed = [Math]::Max($elapsedSec, 0.001)
	}

	$speedup = if ($k1Elapsed -and $elapsedSec -gt 0) {
		[Math]::Round($k1Elapsed / $elapsedSec, 3)
	} else {
		0
	}

	"$k,$mode,$elapsedSec,$samplesPerSec,$speedup,$runLog" | Out-File -FilePath $summaryPath -Append -Encoding utf8
}

Write-Host "Benchmark summary saved to $summaryPath"
