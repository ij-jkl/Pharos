<#
.SYNOPSIS
    One-command setup for Pharos: installs what is missing, then proves the CLI works.

.DESCRIPTION
    Safe to re-run. Every step checks before it acts, so a second run repairs a half-finished
    first one instead of duplicating it. Nothing is installed system-wide except uv, and
    nothing outside the install directory is modified.

.PARAMETER Path
    Where to clone Pharos if you are not already inside a clone. Default: ~\Pharos

.PARAMETER SkipDemo
    Set up and stop, without the closing demonstration run.

.EXAMPLE
    irm https://raw.githubusercontent.com/ij-jkl/Pharos/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -Path D:\code\Pharos
#>

[CmdletBinding()]
param(
    [string] $Path = (Join-Path $HOME 'Pharos'),
    [switch] $SkipDemo
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoUrl = 'https://github.com/ij-jkl/Pharos.git'
$OllamaUrl = 'http://localhost:11434'

# ---------------------------------------------------------------------------------------------
# Output helpers. Steps are numbered so a failure report says which one, and the exact command
# that failed is always echoed rather than summarised.
# ---------------------------------------------------------------------------------------------

$script:StepNo = 0

function Write-Step {
    param([string] $Message)
    $script:StepNo++
    Write-Host ''
    Write-Host ("[{0}] {1}" -f $script:StepNo, $Message) -ForegroundColor Cyan
}

function Write-Ok {
    param([string] $Message)
    Write-Host "    ok  $Message" -ForegroundColor Green
}

function Write-Info {
    param([string] $Message)
    Write-Host "        $Message" -ForegroundColor DarkGray
}

function Write-Note {
    param([string] $Message)
    Write-Host "    !   $Message" -ForegroundColor Yellow
}

function Stop-WithHelp {
    param([string] $Problem, [string] $Fix)
    Write-Host ''
    Write-Host "FAILED: $Problem" -ForegroundColor Red
    if ($Fix) { Write-Host "        $Fix" -ForegroundColor Yellow }
    Write-Host ''
    exit 1
}

function Test-Command {
    param([string] $Name)
    return [bool] (Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------------------------

Write-Host ''
Write-Host '  Pharos — setup' -ForegroundColor White
Write-Host '  See your context budget before your agent silently overflows it.' -ForegroundColor DarkGray

# --- 1. git ----------------------------------------------------------------------------------

Write-Step 'Checking for git'

$insideClone = (Test-Path 'pyproject.toml') -and
               (Select-String -Path 'pyproject.toml' -Pattern '^name\s*=\s*"pharos"' -Quiet -ErrorAction SilentlyContinue)

if ($insideClone) {
    Write-Ok 'already inside a Pharos clone — no download needed'
}
elseif (-not (Test-Command 'git')) {
    Stop-WithHelp 'git is not installed, and it is needed to download Pharos.' `
                  'Install it from https://git-scm.com/download/win (or: winget install Git.Git), then re-run this script.'
}
else {
    Write-Ok 'git found'
}

# --- 2. uv -----------------------------------------------------------------------------------

Write-Step 'Checking for uv (the Python toolchain manager Pharos uses)'

if (Test-Command 'uv') {
    Write-Ok ('uv found — ' + (uv --version))
}
else {
    Write-Info 'not found; installing from astral.sh'
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    catch {
        Stop-WithHelp "the uv installer failed: $($_.Exception.Message)" `
                      'Install it manually from https://docs.astral.sh/uv/ and re-run this script.'
    }

    # The installer updates PATH for future sessions, not this one. Add its default location so
    # the rest of the script can proceed without asking for a new terminal.
    $uvBin = Join-Path $HOME '.local\bin'
    if (Test-Path (Join-Path $uvBin 'uv.exe')) {
        $env:Path = "$uvBin;$env:Path"
    }

    if (-not (Test-Command 'uv')) {
        Stop-WithHelp 'uv installed but is not on PATH in this session.' `
                      'Close this terminal, open a new one, and re-run the script.'
    }
    Write-Ok ('uv installed — ' + (uv --version))
}

# --- 3. the repository -----------------------------------------------------------------------

Write-Step 'Locating Pharos'

if ($insideClone) {
    $root = (Get-Location).Path
    Write-Ok "using the clone you are in: $root"
}
elseif (Test-Path (Join-Path $Path '.git')) {
    $root = (Resolve-Path $Path).Path
    Write-Ok "found an existing clone: $root"
    Write-Info 'leaving it as it is — pull manually if you want the latest'
}
else {
    Write-Info "cloning into $Path"
    git clone --quiet $RepoUrl $Path
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp "git clone failed (exit $LASTEXITCODE)." `
                      "Check network access to github.com, or clone manually: git clone $RepoUrl"
    }
    $root = (Resolve-Path $Path).Path
    Write-Ok "cloned to $root"
}

Set-Location $root

# --- 4. dependencies -------------------------------------------------------------------------

Write-Step 'Installing dependencies (this fetches Python 3.12 the first time — give it a minute)'

uv sync
if ($LASTEXITCODE -ne 0) {
    Stop-WithHelp "uv sync failed (exit $LASTEXITCODE)." `
                  'Scroll up for the resolver error. A compile attempt for llama-cpp-python means the prebuilt CPU wheel index was not used.'
}
Write-Ok 'dependencies installed'

# --- 5. configuration ------------------------------------------------------------------------

Write-Step 'Setting up pharos.toml'

if (Test-Path 'pharos.toml') {
    Write-Ok 'pharos.toml already exists — left untouched'
}
else {
    Copy-Item 'pharos.toml.example' 'pharos.toml'
    Write-Ok 'created pharos.toml from the example'
    Write-Info 'machine-specific and gitignored; edit it whenever you like'
}

# --- 6. the backend --------------------------------------------------------------------------
# Not required to install or to run the pre-flight, but it is the difference between a real
# verdict and an arithmetic exercise, so the state is reported plainly either way.

Write-Step 'Looking for an Ollama backend'

$loadedModel = $null
$haveBackend = $false

try {
    $ps = Invoke-RestMethod -Uri "$OllamaUrl/api/ps" -TimeoutSec 4
    $haveBackend = $true
    if ($ps.models -and $ps.models.Count -gt 0) {
        $loadedModel = $ps.models[0].name
        Write-Ok "backend up, model resident: $loadedModel"
    }
    else {
        Write-Ok 'backend up, but no model is loaded right now'
        try {
            $tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 4
            if ($tags.models -and $tags.models.Count -gt 0) {
                $available = $tags.models[0].name
                Write-Info "you have $($tags.models.Count) model(s) pulled, e.g. $available"
                Write-Info "load one with:  ollama run $available `"hi`""
            }
            else {
                Write-Info 'no models pulled yet — try:  ollama pull qwen3:8b'
            }
        }
        catch {
            Write-Info 'could not list pulled models'
        }
    }
}
catch {
    Write-Note "no backend at $OllamaUrl"
    Write-Info 'Pharos installs and runs fine without one — the pre-flight just cannot give a'
    Write-Info 'real verdict, because the budget it compares against is the window your backend'
    Write-Info 'actually loaded. Install Ollama from https://ollama.com to get verdicts.'
}

# --- 7. does it actually work? ---------------------------------------------------------------

Write-Step 'Verifying the CLI runs'

uv run pharos check --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stop-WithHelp 'the pharos command did not run.' 'Scroll up for the error, and open an issue with it.'
}
Write-Ok 'pharos responds'

if ($SkipDemo) {
    Write-Host ''
    Write-Host '  Setup complete.' -ForegroundColor Green
    Write-Host ''
    exit 0
}

# --- 8. the demonstration --------------------------------------------------------------------
# Run against Pharos's own source, so it works on a fresh clone with nothing else on the
# machine. --target fixes the per-part budget, which makes the run identical with or without a
# backend; drop it once a model is loaded and the real window is used instead.

Write-Step 'Demonstration — a prompt that looks small and is not'

$demoPrompt = 'Refactor everything in `pharos/proxy/` so the error handling is consistent, following `README.md`'

Write-Host ''
Write-Host '    The prompt:' -ForegroundColor White
Write-Host "      $demoPrompt" -ForegroundColor DarkGray
Write-Host ''
Write-Host '    Thirty tokens to read. Now ask what it actually costs:' -ForegroundColor White
Write-Host ''
Write-Host '      uv run pharos check "<the prompt above>" --target 8000' -ForegroundColor DarkCyan
Write-Host ''

uv run pharos check $demoPrompt --target 8000

Write-Host ''
Write-Host '    The FLOOR is what the prompt guarantees; the CEILING is what it can reach if the' -ForegroundColor White
Write-Host '    agent reads every file in the directory you named. When the ceiling is past your' -ForegroundColor White
Write-Host '    window, that is the silent overflow — cut it up instead:' -ForegroundColor White
Write-Host ''
Write-Host '      uv run pharos split "<the same prompt>" --target 8000 --out parts/' -ForegroundColor DarkCyan
Write-Host ''

# Written to a temp directory rather than into the clone: a setup script should not leave
# untracked files behind in a repository the user has just downloaded.
$demoOut = Join-Path $env:TEMP 'pharos-demo-parts'

uv run pharos split $demoPrompt --target 8000 --out $demoOut 2>&1 |
    Select-String -Pattern '^(Split plan|part \d)' |
    ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }

Write-Host ''
Write-Ok "each part fits, and all of them were written to $demoOut"

# --- 9. where to go next ---------------------------------------------------------------------

Write-Host ''
Write-Host '  Setup complete.' -ForegroundColor Green
Write-Host ''
Write-Host "  You are in: $root" -ForegroundColor DarkGray
Write-Host ''
Write-Host '  Check a prompt of your own:' -ForegroundColor White
Write-Host '    uv run pharos check "your prompt here, naming `some/file.py`"' -ForegroundColor DarkCyan
Write-Host ''
Write-Host '  Cut one that does not fit:' -ForegroundColor White
Write-Host '    uv run pharos split "your prompt" --out parts/' -ForegroundColor DarkCyan
Write-Host ''
Write-Host '  Watch context live, and point your coding agent at 127.0.0.1:11435:' -ForegroundColor White
Write-Host '    uv run pharos' -ForegroundColor DarkCyan
Write-Host ''

if (-not $haveBackend) {
    Write-Note 'Install Ollama and load a model to get real verdicts instead of --target arithmetic.'
    Write-Host ''
}
elseif (-not $loadedModel) {
    Write-Note 'Load a model (ollama run <model> "hi"), then drop --target to check against your real window.'
    Write-Host ''
}
