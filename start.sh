#!/usr/bin/env bash
#
# The only command you need: sets Pharos up the first time, opens the CLI every time after.
#
# Run it once and it installs what is missing, configures the rest, and shows a demonstration.
# Run it again and it notices everything is already in place, skips all of that, and takes you
# straight to the prompt.
#
# Setup is re-detected rather than remembered, so a half-finished first run is repaired by a
# second one, and a `git pull` that changes dependencies re-syncs automatically (uv.lock newer
# than the environment is the signal).
#
# Usage:  ./start.sh [--reinstall] [--setup-only] [--path DIR]

set -euo pipefail

REPO_URL="https://github.com/ij-jkl/Pharos.git"
OLLAMA_URL="http://localhost:11434"
FALLBACK_TARGET=8000

INSTALL_PATH="$HOME/Pharos"
REINSTALL=0
SETUP_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --reinstall)  REINSTALL=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --path)       INSTALL_PATH="$2"; shift 2 ;;
        -h|--help)    sed -n '2,15p' "$0"; exit 0 ;;
        *)            echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

# --- output helpers ---------------------------------------------------------------------------

if [ -t 1 ]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_GREY=$'\033[90m'
    C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_OFF=$'\033[0m'
else
    C_CYAN=""; C_GREEN=""; C_GREY=""; C_YELLOW=""; C_RED=""; C_OFF=""
fi

STEP_NO=0
step() { STEP_NO=$((STEP_NO + 1)); printf '\n%s[%d] %s%s\n' "$C_CYAN" "$STEP_NO" "$1" "$C_OFF"; }
ok()   { printf '    %sok  %s%s\n' "$C_GREEN" "$1" "$C_OFF"; }
info() { printf '        %s%s%s\n' "$C_GREY" "$1" "$C_OFF"; }
note() { printf '    %s!   %s%s\n' "$C_YELLOW" "$1" "$C_OFF"; }

# Drawn, not generated: a figlet dependency for five lines of decoration is not worth an install
# step. Quoted with <<'EOF' so the backslashes arrive as themselves.
banner() {
    printf '\n%s' "$C_CYAN"
    cat <<'EOF'
   ____  _   _    _    ____   ___  ____
  |  _ \| | | |  / \  |  _ \ / _ \/ ___|
  | |_) | |_| | / _ \ | |_) | | | \___ \
  |  __/|  _  |/ ___ \|  _ <| |_| |___) |
  |_|   |_| |_/_/   \_\_| \_\\___/|____/
EOF
    printf '%s' "$C_OFF"
}

die() {
    printf '\n%sFAILED: %s%s\n' "$C_RED" "$1" "$C_OFF" >&2
    [ $# -gt 1 ] && printf '        %s%s%s\n' "$C_YELLOW" "$2" "$C_OFF" >&2
    printf '\n' >&2
    exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- locate the clone -------------------------------------------------------------------------

ROOT=""
if [ -f pyproject.toml ] && grep -q '^name = "pharos"' pyproject.toml 2>/dev/null; then
    ROOT="$PWD"
elif [ -f "$INSTALL_PATH/pyproject.toml" ]; then
    ROOT="$(cd "$INSTALL_PATH" && pwd)"
    cd "$ROOT"
fi

# --- is setup already done? -------------------------------------------------------------------
# Three independent facts, so a partial install is detected as partial. The uv.lock comparison
# is what makes `git pull` safe: a lock newer than the environment means dependencies moved.

HAVE_UV=0; have uv && HAVE_UV=1
HAVE_ENV=0
HAVE_CONFIG=0

if [ -n "$ROOT" ]; then
    # bin/ on Unix, Scripts/ under Git Bash on Windows — the same clone can be driven from
    # either shell, and reporting "not installed" against a working environment would re-sync
    # on every single run.
    VENV_MARKER=""
    for candidate in "$ROOT/.venv/bin/pharos" "$ROOT/.venv/Scripts/pharos.exe"; do
        [ -e "$candidate" ] && VENV_MARKER="$candidate" && break
    done
    if [ -n "$VENV_MARKER" ] && [ -f "$ROOT/uv.lock" ]; then
        [ ! "$ROOT/uv.lock" -nt "$VENV_MARKER" ] && HAVE_ENV=1
    fi
    [ -f "$ROOT/pharos.toml" ] && HAVE_CONFIG=1
fi

READY=0
if [ "$HAVE_UV" = 1 ] && [ "$HAVE_ENV" = 1 ] && [ "$HAVE_CONFIG" = 1 ] && [ "$REINSTALL" = 0 ]; then
    READY=1
fi

banner

if [ "$READY" = 1 ]; then
    printf '\n  ready\n'
    printf '  %sAlready set up; skipping installation.%s\n' "$C_GREY" "$C_OFF"
else
    printf '\n  first-time setup\n'
    printf '  %sSee your context budget before your agent silently overflows it.%s\n' "$C_GREY" "$C_OFF"

    if [ -z "$ROOT" ]; then
        step "Checking for git"
        have git || die "git is not installed, and it is needed to download Pharos." \
                        "Install it with your package manager, then re-run."
        ok "git found"

        step "Downloading Pharos into $INSTALL_PATH"
        git clone --quiet "$REPO_URL" "$INSTALL_PATH" \
            || die "git clone failed." "Check network access to github.com, or clone manually: git clone $REPO_URL"
        ROOT="$(cd "$INSTALL_PATH" && pwd)"
        cd "$ROOT"
        ok "cloned to $ROOT"
    fi

    step "Checking for uv (the Python toolchain manager Pharos uses)"
    if [ "$HAVE_UV" = 1 ]; then
        ok "uv found — $(uv --version)"
    else
        info "not found; installing from astral.sh"
        curl -LsSf https://astral.sh/uv/install.sh | sh \
            || die "the uv installer failed." "Install it manually from https://docs.astral.sh/uv/ and re-run."

        # The installer updates the shell profile, not this process.
        export PATH="$HOME/.local/bin:$PATH"
        have uv || die "uv installed but is not on PATH in this session." \
                       "Open a new terminal and re-run."
        ok "uv installed — $(uv --version)"
    fi

    step "Installing dependencies (this fetches Python 3.12 the first time — give it a minute)"
    uv sync || die "uv sync failed." \
                   "Scroll up for the resolver error. A compile attempt for llama-cpp-python means the prebuilt CPU wheel index was not used."
    ok "dependencies installed"

    step "Setting up pharos.toml"
    if [ -f pharos.toml ]; then
        ok "pharos.toml already exists — left untouched"
    else
        cp pharos.toml.example pharos.toml
        ok "created pharos.toml from the example"
        info "machine-specific and gitignored; edit it whenever you like"
    fi
fi

cd "$ROOT"

# --- the backend ------------------------------------------------------------------------------

LOADED_MODEL=""
HAVE_BACKEND=0

detect_loaded_model() {
    LOADED_MODEL=""
    local json
    if json=$(curl -sf -m 4 "$OLLAMA_URL/api/ps" 2>/dev/null); then
        HAVE_BACKEND=1
        LOADED_MODEL=$(printf '%s' "$json" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p' | head -1)
    else
        HAVE_BACKEND=0
    fi
}

# The model named in pharos.toml is the one whose GGUF vocabulary gets loaded, so it is the only
# one that counts as exact. Loading another still works, but every count comes back marked
# untrusted, which is worth saying out loud rather than leaving to be discovered.
configured_model() {
    [ -f "$ROOT/pharos.toml" ] || return 0
    sed -n 's/^[[:space:]]*model[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$ROOT/pharos.toml" | head -1
}

same_model() {
    # /api/ps reports `<name>:latest`; pharos.toml usually holds the untagged name.
    [ -n "$1" ] && [ -n "$2" ] && [ "${1%:latest}" = "${2%:latest}" ]
}

offer_model_load() {
    local tags names sizes configured count i name gb answer choice default
    tags=$(curl -sf -m 4 "$OLLAMA_URL/api/tags" 2>/dev/null) || return 0

    names=$(printf '%s' "$tags" | grep -o '"name":"[^"]*"' | sed 's/"name":"\(.*\)"/\1/')
    [ -z "$names" ] && { note "No models pulled yet. Get one with:  ollama pull qwen3:8b"; return 0; }

    sizes=$(printf '%s' "$tags" | grep -o '"size":[0-9]*' | sed 's/"size"://')
    configured=$(configured_model)

    printf '\n  No model is loaded, so there is no real window to measure against.\n'
    printf '  You already have these:\n\n'

    count=0
    default=1
    while IFS= read -r name; do
        count=$((count + 1))
        gb=$(printf '%s' "$sizes" | sed -n "${count}p")
        gb=$(awk "BEGIN{printf \"%.1f\", ${gb:-0}/1073741824}")
        if same_model "$name" "$configured"; then
            default=$count
            printf '    %2d. %-34s %5s GB   <- named in pharos.toml, counts exactly\n' "$count" "$name" "$gb"
        else
            printf '    %2d. %-34s %5s GB\n' "$count" "$name" "$gb"
        fi
    done <<EOF
$names
EOF

    choice=$(printf '%s' "$names" | sed -n "${default}p")
    printf '\n  %sLoad which? [Enter for %s, a number, or s to skip]%s ' "$C_CYAN" "$choice" "$C_OFF"
    IFS= read -r answer || return 0
    answer=$(printf '%s' "$answer" | tr -d '[:space:]')

    case "$answer" in
        s|skip|n|no) return 0 ;;
        '') ;;
        *[!0-9]*) note "Not one of the listed numbers - skipping."; return 0 ;;
        *)
            if [ "$answer" -lt 1 ] || [ "$answer" -gt "$count" ]; then
                note "Not one of the listed numbers - skipping."; return 0
            fi
            choice=$(printf '%s' "$names" | sed -n "${answer}p") ;;
    esac

    if [ -n "$configured" ] && ! same_model "$choice" "$configured"; then
        note "pharos.toml names '$configured', so counts for '$choice' will be marked untrusted."
    fi

    printf '\n'
    info "loading $choice into VRAM - first load of a large model takes a while"
    curl -sf -m 600 "$OLLAMA_URL/api/chat" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$choice\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" \
        >/dev/null 2>&1 || { note "could not load it"; return 0; }

    detect_loaded_model
    if [ -n "$LOADED_MODEL" ]; then
        ok "loaded $LOADED_MODEL"
    else
        note "it did not stay resident - continuing without a verdict"
    fi
}

detect_loaded_model

if [ "$READY" = 0 ]; then
    step "Looking for an Ollama backend"
    if [ -n "$LOADED_MODEL" ]; then
        ok "backend up, model resident: $LOADED_MODEL"
    elif [ "$HAVE_BACKEND" = 1 ]; then
        ok "backend up, but no model is loaded right now"
    else
        note "no backend at $OLLAMA_URL"
        info "Pharos runs fine without one — the pre-flight just cannot give a real verdict,"
        info "because the budget it compares against is the window your backend actually"
        info "loaded. Install Ollama from https://ollama.com to get verdicts."
    fi

    step "Verifying the CLI runs"
    uv run pharos check --help >/dev/null || die "the pharos command did not run." \
                                                 "Scroll up for the error, and open an issue with it."
    ok "pharos responds"
fi

if [ "$SETUP_ONLY" = 1 ]; then
    printf '\n  %sSetup complete. Run ./start.sh again to open the CLI.%s\n\n' "$C_GREEN" "$C_OFF"
    exit 0
fi

# Nothing resident means every verdict below would be arithmetic against a number nobody
# measured. The models are already on disk, so offer them rather than printing a command to go
# and type somewhere else.
if [ "$HAVE_BACKEND" = 1 ] && [ -z "$LOADED_MODEL" ] && [ -t 0 ]; then
    offer_model_load
fi

# --- the CLI ----------------------------------------------------------------------------------

run_pharos() {
    # Anything after the prompt is forwarded verbatim (--semantic, today). Dropping it
    # would make `s!` silently run a plain split, the one failure this feature must not have.
    local verb="$1" prompt="$2"
    shift 2
    printf '\n'
    if [ -z "$LOADED_MODEL" ]; then
        uv run pharos "$verb" "$prompt" --target "$FALLBACK_TARGET" "$@" || true
        printf '\n'
        info "(no model resident — planned against a stand-in $FALLBACK_TARGET-token window)"
    else
        uv run pharos "$verb" "$prompt" "$@" || true
        printf '\n'
    fi
}

printf '\n'
if [ -n "$LOADED_MODEL" ]; then
    printf '  %sModel: %s — verdicts are against your real loaded window.%s\n' "$C_GREY" "$LOADED_MODEL" "$C_OFF"
else
    printf '  %sNo model resident — using a stand-in %s-token window.%s\n' "$C_GREY" "$FALLBACK_TARGET" "$C_OFF"
    printf '  %sLoad one (ollama run <model> "hi") and re-run for real verdicts.%s\n' "$C_GREY" "$C_OFF"
fi

printf '\n  Type a prompt to pre-flight it. Name files and folders in backticks,\n'
printf '  %se.g.  Refactor everything in `pharos/proxy/` following `README.md`%s\n\n' "$C_GREY" "$C_OFF"
printf '    %ss <prompt>   cut it into parts that fit        d   live dashboard%s\n' "$C_GREY" "$C_OFF"
printf '    %ss! <prompt>  the same, grouped by meaning         q   quit%s\n' "$C_GREY" "$C_OFF"
printf '    %sr <prompt>   carry it out (WRITES files)   r! same, stop at the first break%s\n' "$C_GREY" "$C_OFF"

if [ ! -t 0 ]; then
    printf '\n'
    note "Input is not a terminal, so the prompt is skipped. Run ./start.sh directly to use it."
    printf '\n'
    exit 0
fi

while true; do
    printf '\n%spharos>%s ' "$C_CYAN" "$C_OFF"
    IFS= read -r line || break
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [ -z "$line" ] && continue
    case "$line" in
        q|quit|exit) break ;;
        d|dash|dashboard)
            info "starting the dashboard and proxy — press q in the dashboard to come back"
            uv run pharos || true
            continue ;;
        # Checked before plain `s `, or "s! foo" would split a prompt literally called "! foo".
        s!\ *) run_pharos split "${line#s! }" --semantic; continue ;;
        s\ *) run_pharos split "${line#s }"; continue ;;
        # The only command here that writes to disk, so it says so first. Not routed through
        # run_pharos: a run owns the terminal for minutes and streams its own progress, and
        # capturing that would hide the output that proves it is not stuck.
        # Before plain `r `, for the same reason `s!` is before `s `.
        r!\ *)
            info "carrying out the task — WRITES files, and stops at the first part that breaks one"
            uv run pharos run "${line#r! }" --stop-on-break || true
            continue ;;
        r\ *)
            info "carrying out the task — this WRITES files; git branch or snapshot is your undo"
            uv run pharos run "${line#r }" || true
            continue ;;
    esac

    run_pharos check "$line"
done

printf '\n  %sBye. Run ./start.sh any time — it will skip straight back to here.%s\n\n' "$C_GREEN" "$C_OFF"
