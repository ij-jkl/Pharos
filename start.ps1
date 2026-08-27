<#
.SYNOPSIS
    The only command you need: sets Pharos up the first time, opens the CLI every time after.

.DESCRIPTION
    Run it once and it installs what is missing, configures the rest, and shows a
    demonstration. Run it again and it notices everything is already in place, skips all of
    that, and takes you straight to the prompt.

    Setup is re-detected rather than remembered, so a half-finished first run is repaired by a
    second one, and a `git pull` that changes dependencies re-syncs automatically (uv.lock
    newer than the environment is the signal).

.PARAMETER Path
    Where to clone Pharos if you are not already inside a clone. Default: ~\Pharos

.PARAMETER Reinstall
    Force the setup steps even if everything looks present.

.PARAMETER SetupOnly
    Do the setup and stop, without opening the CLI.

.EXAMPLE
    .\start.ps1

.EXAMPLE
    .\start.ps1 -Reinstall

.NOTES
    KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 is still the default `powershell.exe` on
    Windows 11, and it reads a script without a byte-order mark as the ANSI code page, not as
    UTF-8. A single em dash inside a comment is therefore enough to turn every string after it
    into a parse error the reader cannot connect to anything they did. PowerShell 7 defaults to
    UTF-8 and shows none of this, so it will not be caught by testing on a modern host.
#>

[CmdletBinding()]
param(
    [string] $Path = (Join-Path $HOME 'Pharos'),
    [switch] $Reinstall,
    [switch] $SetupOnly
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# The report is UTF-8 (the floor and ceiling lines lead with an actual >= sign), but a Windows
# PowerShell 5.1 console still reports ibm850 and would decode those bytes as mojibake. Best
# effort: a host that refuses is left alone rather than failing the run over typography.
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$RepoUrl = 'https://github.com/ij-jkl/Pharos.git'
$OllamaUrl = 'http://localhost:11434'
$FallbackTarget = 8000

# ---------------------------------------------------------------------------------------------
# Output helpers
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

function Write-Banner {
    # Drawn, not generated: a figlet dependency for five lines of decoration is not worth an
    # install step, and hard-coding it keeps the file ASCII-clean (see the .NOTES block).
    Write-Host ''
    Write-Host '   ____  _   _    _    ____   ___  ____  ' -ForegroundColor Cyan
    Write-Host '  |  _ \| | | |  / \  |  _ \ / _ \/ ___| ' -ForegroundColor Cyan
    Write-Host '  | |_) | |_| | / _ \ | |_) | | | \___ \ ' -ForegroundColor Cyan
    Write-Host '  |  __/|  _  |/ ___ \|  _ <| |_| |___) |' -ForegroundColor Cyan
    Write-Host '  |_|   |_| |_/_/   \_\_| \_\\___/|____/ ' -ForegroundColor Cyan
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
# Locate the clone first: everything else is relative to it, and whether we are inside one
# decides whether git is needed at all.
# ---------------------------------------------------------------------------------------------

$insideClone = (Test-Path 'pyproject.toml') -and
               (Select-String -Path 'pyproject.toml' -Pattern '^name\s*=\s*"pharos"' -Quiet -ErrorAction SilentlyContinue)

if ($insideClone) {
    $root = (Get-Location).Path
}
elseif (Test-Path (Join-Path $Path 'pyproject.toml')) {
    $root = (Resolve-Path $Path).Path
    Set-Location $root
}
else {
    $root = $null   # nothing yet - the clone step below creates it
}

# ---------------------------------------------------------------------------------------------
# Is setup already done? Three independent facts, so a partial install is detected as partial.
# The uv.lock comparison is what makes `git pull` safe: a lock file newer than the environment
# means dependencies moved and the environment is stale.
# ---------------------------------------------------------------------------------------------

$haveUv = Test-Command 'uv'
$haveEnv = $false
$haveConfig = $false

if ($root) {
    $marker = Join-Path $root '.venv\Scripts\pharos.exe'
    $lock = Join-Path $root 'uv.lock'
    if ((Test-Path $marker) -and (Test-Path $lock)) {
        $haveEnv = (Get-Item $marker).LastWriteTime -ge (Get-Item $lock).LastWriteTime
    }
    $haveConfig = Test-Path (Join-Path $root 'pharos.toml')
}

$ready = $haveUv -and $haveEnv -and $haveConfig -and -not $Reinstall

Write-Banner

if ($ready) {
    Write-Host ''
    Write-Host '  ready' -ForegroundColor White
    Write-Host '  Already set up; skipping installation.' -ForegroundColor DarkGray
}
else {
    Write-Host ''
    Write-Host '  first-time setup' -ForegroundColor White
    Write-Host '  See your context budget before your agent silently overflows it.' -ForegroundColor DarkGray

    # --- git -------------------------------------------------------------------------------

    if (-not $root) {
        Write-Step 'Checking for git'
        if (-not (Test-Command 'git')) {
            Stop-WithHelp 'git is not installed, and it is needed to download Pharos.' `
                          'Install it from https://git-scm.com/download/win (or: winget install Git.Git), then re-run.'
        }
        Write-Ok 'git found'

        Write-Step "Downloading Pharos into $Path"
        git clone --quiet $RepoUrl $Path
        if ($LASTEXITCODE -ne 0) {
            Stop-WithHelp "git clone failed (exit $LASTEXITCODE)." `
                          "Check network access to github.com, or clone manually: git clone $RepoUrl"
        }
        $root = (Resolve-Path $Path).Path
        Set-Location $root
        Write-Ok "cloned to $root"
    }

    # --- uv --------------------------------------------------------------------------------

    Write-Step 'Checking for uv (the Python toolchain manager Pharos uses)'

    if ($haveUv) {
        Write-Ok ('uv found - ' + (uv --version))
    }
    else {
        Write-Info 'not found; installing from astral.sh'
        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        }
        catch {
            Stop-WithHelp "the uv installer failed: $($_.Exception.Message)" `
                          'Install it manually from https://docs.astral.sh/uv/ and re-run.'
        }

        # The installer updates PATH for future sessions, not this one.
        $uvBin = Join-Path $HOME '.local\bin'
        if (Test-Path (Join-Path $uvBin 'uv.exe')) { $env:Path = "$uvBin;$env:Path" }

        if (-not (Test-Command 'uv')) {
            Stop-WithHelp 'uv installed but is not on PATH in this session.' `
                          'Close this terminal, open a new one, and re-run.'
        }
        Write-Ok ('uv installed - ' + (uv --version))
    }

    # --- dependencies ----------------------------------------------------------------------

    Write-Step 'Installing dependencies (this fetches Python the first time - give it a minute)'

    uv sync
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp "uv sync failed (exit $LASTEXITCODE)." `
                      'Scroll up for the resolver error. A compile attempt for llama-cpp-python means the prebuilt CPU wheel index was not used.'
    }
    Write-Ok 'dependencies installed'

    # --- configuration ---------------------------------------------------------------------

    Write-Step 'Setting up pharos.toml'

    if (Test-Path 'pharos.toml') {
        Write-Ok 'pharos.toml already exists - left untouched'
    }
    else {
        Copy-Item 'pharos.toml.example' 'pharos.toml'
        Write-Ok 'created pharos.toml from the example'
        Write-Info 'machine-specific and gitignored; edit it whenever you like'
    }
}

Set-Location $root

# ---------------------------------------------------------------------------------------------
# The backend. Not required, but it decides whether the pre-flight gives a real verdict or
# arithmetic against a stand-in window, so it is checked every run and reported plainly.
# ---------------------------------------------------------------------------------------------

function Get-LoadedModel {
    try {
        $ps = Invoke-RestMethod -Uri "$OllamaUrl/api/ps" -TimeoutSec 4
        $script:haveBackend = $true
        if ($ps.models -and $ps.models.Count -gt 0) { return $ps.models[0].name }
    }
    catch {
        $script:haveBackend = $false
    }
    return $null
}

function Get-PulledModels {
    try {
        $tags = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 4
        if ($tags.models) { return @($tags.models) }
    }
    catch { }
    return @()
}

function Get-ConfiguredModel {
    <#
        The model named in pharos.toml is the one whose GGUF vocabulary gets loaded, so it is
        the only one that produces counts labelled exact. Loading a different one still works,
        but every count comes back red as "untrusted - other model's tokenizer", which is worth
        pointing at rather than letting someone discover it from the colour.
    #>
    $cfg = Join-Path $root 'pharos.toml'
    if (-not (Test-Path $cfg)) { return $null }
    $hit = Select-String -Path $cfg -Pattern '^\s*model\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($hit) { return $hit.Matches[0].Groups[1].Value }
    return $null
}

function Test-SameModel {
    # /api/ps and /api/tags report `<name>:latest`; the natural thing to write in pharos.toml
    # is the untagged name. Compare with the tag stripped from both sides.
    param([string] $A, [string] $B)
    if (-not $A -or -not $B) { return $false }
    return ($A -replace ':latest$', '') -eq ($B -replace ':latest$', '')
}

function Request-ModelLoad {
    <# Offer the models already on disk, and load the chosen one through the API so no ollama
       binary is needed. Returns the newly loaded model name, or $null if nothing was loaded. #>

    $models = Get-PulledModels
    if ($models.Count -eq 0) {
        Write-Note 'No models pulled yet. Get one with:  ollama pull qwen3:8b'
        return $null
    }

    $configured = Get-ConfiguredModel

    Write-Host ''
    Write-Host '  No model is loaded, so there is no real window to measure against.' -ForegroundColor White
    Write-Host '  You already have these:' -ForegroundColor White
    Write-Host ''

    $default = 0
    for ($i = 0; $i -lt $models.Count; $i++) {
        $name = $models[$i].name
        $gb = [math]::Round($models[$i].size / 1GB, 1)
        $mark = ''
        if (Test-SameModel $name $configured) {
            $mark = '   <- named in pharos.toml, counts exactly'
            $default = $i
        }
        Write-Host ("    {0,2}. {1,-34} {2,5} GB{3}" -f ($i + 1), $name, $gb, $mark) -ForegroundColor Gray
    }

    $choice = $models[$default].name
    Write-Host ''
    Write-Host "  Load which? [Enter for $choice, a number, or s to skip] " -ForegroundColor Cyan -NoNewline

    try { $answer = (Read-Host).Trim() } catch { Write-Host ''; return $null }

    if ($answer -in @('s', 'skip', 'n', 'no')) { return $null }
    if ($answer -ne '') {
        $n = 0
        if (-not [int]::TryParse($answer, [ref]$n) -or $n -lt 1 -or $n -gt $models.Count) {
            Write-Note "Not one of the listed numbers - skipping."
            return $null
        }
        $choice = $models[$n - 1].name
    }

    if ($configured -and -not (Test-SameModel $choice $configured)) {
        Write-Note "pharos.toml names '$configured', so counts for '$choice' will be marked untrusted."
    }

    Write-Host ''
    Write-Info "loading $choice into VRAM - first load of a large model takes a while"

    $body = @{
        model    = $choice
        messages = @(@{ role = 'user'; content = 'hi' })
        stream   = $false
    } | ConvertTo-Json -Depth 5

    try {
        $null = Invoke-RestMethod -Uri "$OllamaUrl/api/chat" -Method Post -Body $body `
                                  -ContentType 'application/json' -TimeoutSec 600
    }
    catch {
        Write-Note "could not load it: $($_.Exception.Message)"
        return $null
    }

    $now = Get-LoadedModel
    if ($now) { Write-Ok "loaded $now" } else { Write-Note 'it did not stay resident - continuing without a verdict' }
    return $now
}

$haveBackend = $false
$loadedModel = Get-LoadedModel

if (-not $ready) {
    Write-Step 'Looking for an Ollama backend'
    if ($loadedModel) {
        Write-Ok "backend up, model resident: $loadedModel"
    }
    elseif ($haveBackend) {
        Write-Ok 'backend up, but no model is loaded right now'
    }
    else {
        Write-Note "no backend at $OllamaUrl"
        Write-Info 'Pharos runs fine without one - the pre-flight just cannot give a real'
        Write-Info 'verdict, because the budget it compares against is the window your backend'
        Write-Info 'actually loaded. Install Ollama from https://ollama.com to get verdicts.'
    }

    Write-Step 'Verifying the CLI runs'
    uv run pharos check --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp 'the pharos command did not run.' 'Scroll up for the error, and open an issue with it.'
    }
    Write-Ok 'pharos responds'
}

if ($SetupOnly) {
    Write-Host ''
    Write-Host '  Setup complete. Run .\start.ps1 again to open the CLI.' -ForegroundColor Green
    Write-Host ''
    exit 0
}

# Nothing resident means every verdict below would be arithmetic against a number nobody
# measured. The models are already on disk, so offer them rather than printing a command to go
# and type somewhere else.
if ($haveBackend -and -not $loadedModel -and -not [Console]::IsInputRedirected) {
    $loadedModel = Request-ModelLoad
}

# ---------------------------------------------------------------------------------------------
# The CLI. A prompt is checked against the live window when a model is resident; without one
# there is no real budget, so a stand-in is used and labelled as such rather than pretending.
# ---------------------------------------------------------------------------------------------

# The exit code is handed back through a script-scoped variable, and the call sites must NOT
# assign this function's result. Anything a function writes becomes its return value in
# PowerShell, so `$code = Invoke-Check ...` would capture the whole report into $code and print
# nothing - and capturing it would also hand Rich a pipe instead of the console, costing the
# colour and the full terminal width.
$script:LastCheckCode = 0

function Invoke-Check {
    param([string] $Prompt, [switch] $Split, [switch] $Semantic)

    $verb = if ($Split) { 'split' } else { 'check' }
    $cliArgs = @($verb, $Prompt)
    if (-not $loadedModel) { $cliArgs += @('--target', $FallbackTarget) }
    if ($Semantic) { $cliArgs += '--semantic' }

    Write-Host ''
    uv run pharos @cliArgs
    $script:LastCheckCode = $LASTEXITCODE
    Write-Host ''

    if (-not $loadedModel) {
        Write-Info "(no model resident - planned against a stand-in $FallbackTarget-token window)"
    }
}

Write-Host ''
if ($loadedModel) {
    Write-Host "  Model: $loadedModel - verdicts are against your real loaded window." -ForegroundColor DarkGray
}
else {
    Write-Host "  No model resident - using a stand-in $FallbackTarget-token window." -ForegroundColor DarkGray
    Write-Host '  Load one (ollama run <model> "hi") and re-run for real verdicts.' -ForegroundColor DarkGray
}

Write-Host ''
Write-Host '  Type a prompt to pre-flight it. Name files and folders in backticks,' -ForegroundColor White
Write-Host '  e.g.  Refactor everything in `pharos/proxy/` following `README.md`' -ForegroundColor DarkGray
Write-Host ''
Write-Host '    s <prompt>   cut it into parts that fit        d   live dashboard' -ForegroundColor DarkGray
Write-Host '    s! <prompt>  the same, grouped by meaning         q   quit' -ForegroundColor DarkGray
Write-Host '    r <prompt>   carry it out (WRITES files)' -ForegroundColor DarkGray
Write-Host '    r! <prompt>  the same, stopping at the first part that breaks a file' -ForegroundColor DarkGray
Write-Host '    r? <prompt>  the same, and review the diff afterwards (an opinion)' -ForegroundColor DarkGray
Write-Host ''
Write-Host '    pharos run also takes --compact, --reserve-reads and --no-audit' -ForegroundColor DarkGray

if ([Console]::IsInputRedirected) {
    Write-Host ''
    Write-Note 'Input is not a terminal, so the prompt is skipped. Run .\start.ps1 directly to use it.'
    Write-Host ''
    exit 0
}

while ($true) {
    Write-Host ''
    Write-Host 'pharos> ' -ForegroundColor Cyan -NoNewline

    # -NonInteractive hosts throw here rather than returning anything, and that is a normal way
    # to be run (a CI step, a wrapper script), not a failure worth a stack trace.
    try {
        $line = Read-Host
    }
    catch {
        Write-Host ''
        Write-Note 'This host cannot prompt (running non-interactively). Use: uv run pharos check "<prompt>"'
        break
    }

    if ($null -eq $line) { break }
    $line = $line.Trim()

    if ($line -eq '') { continue }
    if ($line -in @('q', 'quit', 'exit')) { break }

    if ($line -in @('d', 'dash', 'dashboard')) {
        Write-Info 'starting the dashboard and proxy - press q in the dashboard to come back'
        uv run pharos
        continue
    }

    # Matched before plain 's ', or "s! foo" would split a prompt literally called "! foo".
    if ($line -match '^s!\s+(.+)$') {
        Invoke-Check -Prompt $Matches[1] -Split -Semantic
        continue
    }

    if ($line -match '^s\s+(.+)$') {
        Invoke-Check -Prompt $Matches[1] -Split
        continue
    }

    # The only command here that changes anything on disk, so it says so before it starts.
    # Not routed through Invoke-Check: a run owns the terminal for minutes and streams its own
    # progress, and capturing that would hide the very output that proves it is not stuck.
    # Before plain 'r ', for the same reason 's!' is before 's '.
    if ($line -match '^r!\s+(.+)$') {
        Write-Info 'carrying out the task - WRITES files, and stops at the first part that breaks one'
        uv run pharos run $Matches[1] --stop-on-break
        continue
    }

    # 'r?' before plain 'r ', same reason again. The review runs after the verdict and
    # cannot change it; see pharos/agent/review.py.
    if ($line -match '^r\?\s+(.+)$') {
        Write-Info 'carrying out the task - WRITES files, then asks the model what it thinks of the diff'
        uv run pharos run $Matches[1] --review
        continue
    }

    if ($line -match '^r\s+(.+)$') {
        Write-Info 'carrying out the task - this WRITES files; git branch or snapshot is your undo'
        uv run pharos run $Matches[1]
        continue
    }

    Invoke-Check -Prompt $line
    if ($script:LastCheckCode -eq 1) {
        Write-Note 'That does not fit. Prefix the same prompt with "s " to cut it into parts that do.'
    }
}

Write-Host ''
Write-Host '  Bye. Run .\start.ps1 any time - it will skip straight back to here.' -ForegroundColor Green
Write-Host ''
