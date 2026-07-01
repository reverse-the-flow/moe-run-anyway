param(
    [switch]$PreflightOnly,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\jpret\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Runner = Join-Path $RepoRoot "scripts\run_live_baseline.py"
$SuitePath = Join-Path $RepoRoot "memory-moe-mvp\data\mixtral_probe_prompts.json"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutputRoot = Join-Path $RepoRoot "memory-moe-mvp\runtime-probe-runs\pc-ollama-$RunStamp"
$LogPath = Join-Path $OutputRoot "pc-ollama-phase2.log"
$ResultPath = Join-Path $OutputRoot "pc-ollama-phase2-results.jsonl"
$PlanPath = Join-Path $OutputRoot "pc-ollama-phase2-plan.json"
$TagsPath = Join-Path $OutputRoot "ollama-tags.json"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Write-RunLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $LogPath -Append | Out-Host
}

$Models = @(
    [ordered]@{
        model = "dolphin-mixtral:8x7b"
        label = "pc-ollama-dolphin-mixtral-8x7b"
        family = "mixtral_style"
        request_max_tokens = 256
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "qwen3-coder:30b"
        label = "pc-ollama-qwen3-coder-30b"
        family = "qwen3moe"
        request_max_tokens = 256
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q3_K_S"
        label = "pc-ollama-qwen3-coder-30b-a3b-q3ks"
        family = "qwen3moe"
        request_max_tokens = 256
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "qwen3.6:35b"
        label = "pc-ollama-qwen36-35b"
        family = "qwen35moe"
        request_max_tokens = 256
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "nemotron-3-nano:30b-a3b-q4_K_M"
        label = "pc-ollama-nemotron-3-nano-30b-a3b"
        family = "nemotron_h_moe"
        request_max_tokens = 512
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M"
        label = "pc-ollama-nemotron-cascade-2-30b-a3b"
        family = "nemotron_h_moe"
        request_max_tokens = 512
        timeout_seconds = 3600
    },
    [ordered]@{
        model = "glm-4.7-flash:Q8_0"
        label = "pc-ollama-glm-47-flash-q8"
        family = "glm4moelite"
        request_max_tokens = 256
        timeout_seconds = 3600
    }
)

$SkippedModels = @(
    [ordered]@{
        model = "hf.co/unsloth/DeepSeek-V3.1-GGUF:TQ1_0"
        reason = "170 GB local model; skip initial PC architecture smoke to avoid monopolizing the machine."
    },
    [ordered]@{
        model = "llama4:17b-scout-16e-instruct-q4_K_M"
        reason = "67 GB local model; defer until the smaller MoE-ish PC run finishes cleanly."
    },
    [ordered]@{
        model = "qwen3-coder-next:Q4_K_M"
        reason = "51 GB local model; defer until baseline runtime and artifact shape are proven on smaller targets."
    }
)

$Plan = [ordered]@{
    created_at = (Get-Date).ToString("o")
    mode = "pc_ollama_phase2"
    preflight_only = [bool]$PreflightOnly
    smoke_test = [bool]$SmokeTest
    base_url = "http://127.0.0.1:11434"
    backend_family = "ollama_openai_compatible"
    output_root = $OutputRoot
    models = $Models
    skipped_models = $SkippedModels
    notes = @(
        "Uses Docker Ollama through the local OpenAI-compatible HTTP endpoint.",
        "Captures runtime/request evidence only; does not claim semantic expert ids.",
        "Runs one prompt per model from memory-moe-mvp/data/mixtral_probe_prompts.json.",
        "SmokeTest mode captures inventory, preflights one target, and exits without prompt traffic."
    )
}
$Plan | ConvertTo-Json -Depth 8 | Set-Content -Path $PlanPath -Encoding UTF8

Write-RunLog "Starting PC Ollama phase-2 run. OutputRoot=$OutputRoot PreflightOnly=$PreflightOnly SmokeTest=$SmokeTest"

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null
    Write-RunLog "Ollama HTTP endpoint is reachable."
}
catch {
    Write-RunLog "Ollama HTTP endpoint is not reachable yet: $($_.Exception.Message)"
    Write-RunLog "Trying to start Docker container 'ollama' once."
    try {
        & docker start ollama 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Host
        Start-Sleep -Seconds 10
    }
    catch {
        Write-RunLog "Docker start attempt failed: $($_.Exception.Message)"
    }
}

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 30 |
        ConvertTo-Json -Depth 10 |
        Set-Content -Path $TagsPath -Encoding UTF8
    Write-RunLog "Captured Ollama model inventory to $TagsPath"
}
catch {
    Write-RunLog "Failed to capture Ollama model inventory: $($_.Exception.Message)"
}

$ModelsToRun = $Models
if ($SmokeTest) {
    $ModelsToRun = @($Models[0])
    Write-RunLog "SmokeTest mode enabled; only preflighting model=$($ModelsToRun[0].model)"
}

foreach ($item in $ModelsToRun) {
    Write-RunLog "Starting model=$($item.model) label=$($item.label)"

    $RunnerArgs = @(
        $Runner,
        "--base-url", "http://127.0.0.1:11434",
        "--backend-family", "ollama_openai_compatible",
        "--model", $item.model,
        "--output-dir", $OutputRoot,
        "--label", $item.label,
        "--suite-path", $SuitePath,
        "--max-prompts", "1",
        "--repeats", "1",
        "--request-max-tokens", [string]$item.request_max_tokens,
        "--preflight-timeout-seconds", "5",
        "--timeout-seconds", [string]$item.timeout_seconds,
        "--json"
    )

    if ($PreflightOnly -or $SmokeTest) {
        $RunnerArgs += "--preflight-only"
    }

    $StartedAt = Get-Date
    & $Python @RunnerArgs 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Host
    $ExitCode = $LASTEXITCODE
    $CompletedAt = Get-Date

    [ordered]@{
        model = $item.model
        label = $item.label
        family = $item.family
        started_at = $StartedAt.ToString("o")
        completed_at = $CompletedAt.ToString("o")
        exit_code = $ExitCode
        preflight_only = [bool]$PreflightOnly
        smoke_test = [bool]$SmokeTest
    } | ConvertTo-Json -Depth 4 -Compress | Add-Content -Path $ResultPath -Encoding UTF8

    Write-RunLog "Completed model=$($item.model) exit_code=$ExitCode"
}

Write-RunLog "Finished PC Ollama phase-2 run. Results=$ResultPath"
exit 0
