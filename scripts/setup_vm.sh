#!/usr/bin/env bash
#
# Bootstrap a Debian / Ubuntu GCP VM (or any similar host) with the
# OS packages + `uv` needed to run the meds-mcp experiments.
#
# Idempotent: rerunning it is safe; existing installs are detected
# and skipped.
#
# Usage:
#   bash scripts/setup_vm.sh
#
# After this script finishes, open a new shell (or `source ~/.bashrc`)
# so the updated PATH takes effect, then:
#   cd <repo>
#   cp .env.example .env && $EDITOR .env
#   uv sync
#
set -euo pipefail

log()  { printf '\033[1;34m[setup_vm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup_vm]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup_vm]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. OS packages -----------------------------------------------------------
APT_PKGS=(build-essential git curl ca-certificates)

install_apt() {
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found — skipping OS package install (not Debian/Ubuntu?)"
        return
    fi
    local missing=()
    for pkg in "${APT_PKGS[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        log "apt packages already installed: ${APT_PKGS[*]}"
        return
    fi
    log "installing apt packages: ${missing[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends "${missing[@]}"
}

# --- 2. uv --------------------------------------------------------------------
install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv already installed: $(uv --version)"
        return
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        log "uv present at ~/.local/bin/uv but not on PATH yet"
        return
    fi
    log "installing uv (user-local, no sudo)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
}

# --- 3. PATH ------------------------------------------------------------------
# Make ~/.local/bin available to this shell AND future shells via .bashrc.
persist_path() {
    local line='export PATH="$HOME/.local/bin:$PATH"'
    export PATH="$HOME/.local/bin:$PATH"

    local rc="$HOME/.bashrc"
    touch "$rc"
    if grep -qxF "$line" "$rc"; then
        log "~/.local/bin already on PATH in $rc"
    else
        log "appending ~/.local/bin to PATH in $rc"
        printf '\n# added by meds-mcp scripts/setup_vm.sh\n%s\n' "$line" >> "$rc"
    fi
}

# --- 4. Verify ----------------------------------------------------------------
verify() {
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after install"
    log "uv version : $(uv --version)"
    log "git version: $(git --version)"
    log "setup complete — open a new shell or 'source ~/.bashrc' to refresh PATH."
}

install_apt
install_uv
persist_path
verify
