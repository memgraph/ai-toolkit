#Requires -Version 5.1
#
# Install Context Graph for your coding agents (Windows / PowerShell). The
# counterpart of install.sh: this script only gets Python's tooling in place;
# `memgraph-context-graph setup` does the rest: it finds or starts Memgraph,
# wires every agent it finds, sets your identity and checks the result. On a
# console it is interactive; otherwise it prints its plan and runs it.
#
# Usage (PowerShell 5.1 or PowerShell 7+):
#   iwr https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.ps1 -useb | iex
#   .\install.ps1 codex --memgraph-url bolt://my-host:7687     # any `setup` flags
#
# `iex` cannot pass arguments to a piped script, so there the setup flags come from
# CONTEXT_GRAPH_SETUP_ARGS (split on spaces), e.g.
#   $env:CONTEXT_GRAPH_SETUP_ARGS = 'codex --user-id me'; iwr <url> -useb | iex
#
# Undo with `memgraph-context-graph uninstall` (`--all` removes the graph, config and tool too).

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

function Fail([string]$Message) {
    [Console]::Error.WriteLine("FAIL $Message")
    exit 1
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Run a native command and stop on a non-zero exit. Windows PowerShell 5.1 turns any stderr output of a
# native command into a terminating error under $ErrorActionPreference = 'Stop' (even with 2>$null), and
# uv writes routine notices to stderr, so errors are relaxed around the call and the exit code decides.
function Invoke-Native([string]$Command, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command @Arguments } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) {
        Fail "$Command $($Arguments -join ' ') exited with $LASTEXITCODE"
    }
}

# Run a native command for its output only; a failure or stderr noise is not an error.
function Get-NativeOutput([string]$Command, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { return (& $Command @Arguments 2>$null) } catch { return $null } finally { $ErrorActionPreference = $previous }
}

$SetupArgs = @($args)
if ($SetupArgs.Count -eq 0 -and $env:CONTEXT_GRAPH_SETUP_ARGS) {
    $SetupArgs = @($env:CONTEXT_GRAPH_SETUP_ARGS -split '\s+' | Where-Object { $_ })
}

if (-not (Test-Command 'uv')) {
    Write-Host "Installing uv (it manages Python for the memgraph-context-graph tool)"
    # Bypass applies to this child process only; the user's execution policy is left alone.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) { Fail "uv's installer exited with $LASTEXITCODE" }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Test-Command 'uv')) {
        Fail "uv was installed but isn't on PATH; open a new terminal and run this again."
    }
}

# The same set setup checks for (memgraph_context_graph/setup/tool.py) and the plugins' bootstrap installs.
# The viewer is installed here, not left to setup: setup would add it by running `uv tool install` from
# inside the tool's own environment, and Windows won't let uv replace the running memgraph-context-graph.exe.
$Packages = @(
    'tool', 'install', '--quiet', '--python', '3.12', 'memgraph-context-graph[mcp]>=0.10.0',
    '--with', 'skills-graph[memgraph-context-graph]>=0.3.0',
    '--with', 'actions-graph[memgraph-context-graph]>=0.7.0',
    '--with', 'sessions-graph[memgraph-context-graph]>=0.11.0'
)
if ($SetupArgs -notcontains '--no-viewer') { $Packages += @('--with', 'context-graph-viewer>=0.1.1') }
Write-Host "Installing memgraph-context-graph with skills-graph, actions-graph and sessions-graph"
Invoke-Native 'uv' ($Packages + '--upgrade')

# uv's tool shims live in `uv tool dir --bin`, which may not be on PATH yet. Add it to this session, and
# persist it in the user's PATH so a new terminal finds the tool too.
$ToolBin = Get-NativeOutput 'uv' @('tool', 'dir', '--bin')
if (-not $ToolBin) { $ToolBin = "$env:USERPROFILE\.local\bin" }
if (($env:Path -split ';') -notcontains $ToolBin) { $env:Path = "$ToolBin;$env:Path" }
Get-NativeOutput 'uv' @('tool', 'update-shell') | Out-Null

if (-not (Test-Command 'memgraph-context-graph')) {
    Fail "memgraph-context-graph was installed but isn't on PATH; add $ToolBin to PATH and run this again."
}

# Setup reports through its exit code; its stderr (warnings) must not abort a Windows PowerShell 5.1 script.
$ErrorActionPreference = 'Continue'

# Setup's terminal UI needs a console on both ends; without one it runs its plain, non-interactive path.
if (-not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected) {
    & memgraph-context-graph setup @SetupArgs
} else {
    & memgraph-context-graph setup --no-tui @SetupArgs
}
exit $LASTEXITCODE
