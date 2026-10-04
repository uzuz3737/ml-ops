[CmdletBinding()]
param(
    [string]$Config = 'configs/project.yaml',
    [switch]$InfrastructureOnly,
    [ValidateRange(30, 7200)][int]$TimeoutSeconds = 7200,
    [string]$RunId = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousConfig = $env:PIPELINE_CONFIG
Push-Location $projectRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker is required. Install Docker Desktop with Linux containers and start the engine.'
    }
    function Invoke-Docker {
        param([string[]]$DockerArgs)
        & docker @DockerArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Docker command failed (exit $LASTEXITCODE). See the output above."
        }
    }
    $engineType = & docker info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable; start Docker Desktop first.' }
    if ($engineType -ne 'linux') { throw 'This project requires Docker Linux containers.' }
    Invoke-Docker @('compose', 'version')
    if (-not (Test-Path -LiteralPath '.env' -PathType Leaf)) {
        throw 'Create .env from .env.example and set local credentials before running this script.'
    }
    $configAbsolute = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Config))
    $configDirectory = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'configs')) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $configAbsolute.StartsWith($configDirectory, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $configAbsolute -PathType Leaf) -or
        [System.IO.Path]::GetExtension($configAbsolute) -notin @('.yaml', '.yml')) {
        throw 'Config must name an existing YAML file inside this repository''s configs directory.'
    }
    $relativeConfig = $configAbsolute.Substring($projectRoot.Length + 1).Replace('\', '/')
    $env:PIPELINE_CONFIG = $relativeConfig
    foreach ($directory in @('data', 'artifacts', 'schemas', 'examples')) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    Invoke-Docker @('compose', 'config', '--quiet')
    Invoke-Docker @('compose', 'build', 'pipeline-worker', 'airflow-init', 'airflow-webserver', 'airflow-scheduler')
    # Start the worker first so its environment can diagnose missing handoffs.
    Invoke-Docker @('compose', 'up', '-d', '--wait', '--wait-timeout', '300', 'pipeline-worker')
    if (-not $InfrastructureOnly) {
        Invoke-Docker @('compose', 'exec', '-T', 'pipeline-worker', 'python', '-m',
            'mlops_project.pipelines.preflight', '--config', $relativeConfig, '--project-root', '/workspace')
    }
    Invoke-Docker @('compose', 'up', '-d', '--wait', '--wait-timeout', '300',
        'airflow-webserver', 'airflow-scheduler', 'prometheus', 'grafana', 'pipeline-worker')
    if ($InfrastructureOnly) {
        Write-Host 'Infrastructure is running. No training DAG was triggered and no model was deployed.'
        exit 0
    }
    Invoke-Docker @('compose', '--profile', 'application', 'build', 'api')
    Invoke-Docker @('compose', '--profile', 'application', 'up', '-d', '--wait', '--wait-timeout', '300', 'api')
    $launchArgs = @('compose', 'exec', '-T', 'pipeline-worker', 'python',
        '/workspace/scripts/airflow-run.py', '--config', $relativeConfig,
        '--timeout-seconds', $TimeoutSeconds.ToString())
    if ($RunId) { $launchArgs += @('--run-id', $RunId) }
    Invoke-Docker $launchArgs
}
catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
finally {
    $env:PIPELINE_CONFIG = $previousConfig
    Pop-Location
}
