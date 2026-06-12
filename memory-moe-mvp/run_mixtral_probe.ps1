[CmdletBinding()]
param(
    [string]$SuitePath = ".\data\mixtral_probe_prompts.json",
    [string]$Endpoint = "http://127.0.0.1:18080/v1/chat/completions",
    [string]$OutputRoot = ".\probe-results",
    [string]$RunDir = "",
    [int]$Repeats = 2,
    [int]$PauseMilliseconds = 250,
    [int]$MaxPrompts = 0,
    [string[]]$FamilyIds = @(),
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-RunDirectory {
    param([string]$Root)

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $path = Join-Path $Root "mixtral-probe-$timestamp"
    New-Item -ItemType Directory -Force -Path $path | Out-Null
    return $path
}

function Get-PromptPreview {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $squashed = ($Text -replace "\s+", " ").Trim()
    if ($squashed.Length -le 140) {
        return $squashed
    }

    return $squashed.Substring(0, 137) + "..."
}

function Invoke-ProbeRequest {
    param(
        [string]$Uri,
        [hashtable]$Body
    )

    $json = $Body | ConvertTo-Json -Depth 12
    $started = Get-Date
    $response = Invoke-RestMethod -Uri $Uri -Method Post -ContentType "application/json" -Body $json -TimeoutSec 600
    $elapsed = ((Get-Date) - $started).TotalMilliseconds

    return @{
        response = $response
        elapsed_ms = [math]::Round($elapsed, 3)
    }
}

function Write-ProbeSummary {
    param(
        [System.Collections.IList]$Results,
        [string]$RunDirectory,
        [string]$SummaryFilePath
    )

    $summaryRows = @(
        $Results |
            Group-Object family_id |
            ForEach-Object {
                $group = @($_.Group)
                $successes = @($group | Where-Object { $_.finish_reason -ne "error" })
                [pscustomobject]@{
                    family_id = $_.Name
                    runs = $group.Count
                    errors = @($group | Where-Object { $_.finish_reason -eq "error" }).Count
                    mean_elapsed_ms = if ($successes.Count -gt 0) { [math]::Round((($successes | Measure-Object -Property elapsed_ms -Average).Average), 3) } else { $null }
                    mean_prompt_tokens = if ($successes.Count -gt 0) { [math]::Round((($successes | Measure-Object -Property prompt_tokens -Average).Average), 3) } else { $null }
                    mean_completion_tokens = if ($successes.Count -gt 0) { [math]::Round((($successes | Measure-Object -Property completion_tokens -Average).Average), 3) } else { $null }
                    mean_prompt_ms = if ($successes.Count -gt 0) { [math]::Round((($successes | Measure-Object -Property prompt_ms -Average).Average), 3) } else { $null }
                    mean_predicted_ms = if ($successes.Count -gt 0) { [math]::Round((($successes | Measure-Object -Property predicted_ms -Average).Average), 3) } else { $null }
                }
            }
    )

    $summary = [ordered]@{
        created_at = (Get-Date).ToString("o")
        run_dir = $RunDirectory
        total_runs = $Results.Count
        successful_runs = @($Results | Where-Object { $_.finish_reason -ne "error" }).Count
        failed_runs = @($Results | Where-Object { $_.finish_reason -eq "error" }).Count
        family_summaries = $summaryRows
    }

    $summary | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $SummaryFilePath
}

$suite = Get-Content $SuitePath | ConvertFrom-Json
$selectedFamilies = @($suite.families)
if ($FamilyIds.Count -gt 0) {
    $selectedFamilies = @($suite.families | Where-Object { $FamilyIds -contains $_.family_id })
}

$selectedPrompts = @()
foreach ($family in $selectedFamilies) {
    foreach ($prompt in $family.prompts) {
        $selectedPrompts += [pscustomobject]@{
            family_id = $family.family_id
            routing_hypothesis = $family.routing_hypothesis
            probe_id = $prompt.probe_id
            title = $prompt.title
            expected_surface_features = @($prompt.expected_surface_features)
            messages = @($prompt.messages)
        }
    }
}

if ($MaxPrompts -gt 0) {
    $selectedPrompts = @($selectedPrompts | Select-Object -First $MaxPrompts)
}

if ($selectedPrompts.Count -eq 0) {
    throw "No prompts selected from suite."
}

$runDir = $RunDir
if ([string]::IsNullOrWhiteSpace($runDir)) {
    $runDir = New-RunDirectory -Root $OutputRoot
}
else {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
}
$resultsPath = Join-Path $runDir "results.json"
$summaryPath = Join-Path $runDir "summary.json"
$manifestPath = Join-Path $runDir "manifest.json"

if (-not (Test-Path $manifestPath)) {
    $manifest = [ordered]@{
        created_at = (Get-Date).ToString("o")
        suite_name = $suite.suite_name
        target_model = $suite.target_model
        endpoint = $Endpoint
        repeats = $Repeats
        pause_milliseconds = $PauseMilliseconds
        suite_path = $SuitePath
        selected_family_ids = @($selectedFamilies | ForEach-Object { $_.family_id })
        selected_prompt_count = $selectedPrompts.Count
        default_request = $suite.default_request
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $manifestPath
}

$results = New-Object System.Collections.ArrayList
if ($Resume -and (Test-Path $resultsPath)) {
    $existing = Get-Content $resultsPath | ConvertFrom-Json
    foreach ($item in @($existing)) {
        [void]$results.Add($item)
    }
}

$completed = @{}
foreach ($item in $results) {
    $completed["$($item.probe_id)|$($item.repeat)"] = $true
}

foreach ($prompt in $selectedPrompts) {
    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
        $completionKey = "$($prompt.probe_id)|$repeat"
        if ($Resume -and $completed.ContainsKey($completionKey)) {
            Write-Host ("Skipping {0} repeat {1}/{2} (already complete)" -f $prompt.probe_id, $repeat, $Repeats)
            continue
        }

        Write-Host ("Running {0} repeat {1}/{2}" -f $prompt.probe_id, $repeat, $Repeats)

        $body = @{
            model = "dolphin-mixtral"
            messages = $prompt.messages
            temperature = $suite.default_request.temperature
            top_p = $suite.default_request.top_p
            max_tokens = $suite.default_request.max_tokens
            stream = $suite.default_request.stream
        }

        try {
            $invocation = Invoke-ProbeRequest -Uri $Endpoint -Body $body
            $response = $invocation.response

            $messageText = ""
            if ($null -ne $response.choices -and $response.choices.Count -gt 0) {
                $choice = $response.choices[0]
                if ($null -ne $choice.message -and $null -ne $choice.message.content) {
                    $messageText = [string]$choice.message.content
                }
            }

            [void]$results.Add([pscustomobject]@{
                family_id = $prompt.family_id
                probe_id = $prompt.probe_id
                title = $prompt.title
                repeat = $repeat
                elapsed_ms = $invocation.elapsed_ms
                prompt_preview = Get-PromptPreview -Text ([string]$prompt.messages[0].content)
                response_preview = Get-PromptPreview -Text $messageText
                finish_reason = if ($response.choices.Count -gt 0) { $response.choices[0].finish_reason } else { $null }
                prompt_tokens = if ($null -ne $response.usage) { $response.usage.prompt_tokens } else { $null }
                completion_tokens = if ($null -ne $response.usage) { $response.usage.completion_tokens } else { $null }
                total_tokens = if ($null -ne $response.usage) { $response.usage.total_tokens } else { $null }
                prompt_ms = if ($null -ne $response.timings) { $response.timings.prompt_ms } else { $null }
                predicted_ms = if ($null -ne $response.timings) { $response.timings.predicted_ms } else { $null }
                content = $messageText
                expected_surface_features = @($prompt.expected_surface_features)
                routing_hypothesis = $prompt.routing_hypothesis
            })
        }
        catch {
            [void]$results.Add([pscustomobject]@{
                family_id = $prompt.family_id
                probe_id = $prompt.probe_id
                title = $prompt.title
                repeat = $repeat
                elapsed_ms = $null
                prompt_preview = Get-PromptPreview -Text ([string]$prompt.messages[0].content)
                response_preview = ""
                finish_reason = "error"
                prompt_tokens = $null
                completion_tokens = $null
                total_tokens = $null
                prompt_ms = $null
                predicted_ms = $null
                content = ""
                expected_surface_features = @($prompt.expected_surface_features)
                routing_hypothesis = $prompt.routing_hypothesis
                error = $_.Exception.Message
            })
        }

        $results | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $resultsPath
        Write-ProbeSummary -Results $results -RunDirectory $runDir -SummaryFilePath $summaryPath

        Start-Sleep -Milliseconds $PauseMilliseconds
    }
}

Write-ProbeSummary -Results $results -RunDirectory $runDir -SummaryFilePath $summaryPath

Write-Host "Probe run complete."
Write-Host "Manifest: $manifestPath"
Write-Host "Results:  $resultsPath"
Write-Host "Summary:  $summaryPath"
