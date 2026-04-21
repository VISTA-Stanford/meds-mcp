#!/usr/bin/env bash
#
# Bootstrap a Debian / Ubuntu GCP VM for the meds-mcp experiments.
#
# Installs OS packages + uv, optionally copies the cohort CSV + XML corpus
# from GCS to local disk, and generates a .env at the repo root that points
# at the local copies. After the script finishes, you only need to paste
# VAULT_SECRET_KEY into .env and run `uv sync`.
#
# Idempotent: existing installs, files, and mounts are detected and skipped.
#
# Usage:
#   bash scripts/setup_vm.sh                 # uv + apt pkgs only
#   bash scripts/setup_vm.sh --copy-gcs      # also copy CSV + corpus locally
#                                            # and write .env (RECOMMENDED)
#   bash scripts/setup_vm.sh --mount-gcs     # alternative: gcsfuse mount
#                                            # instead of copying
#
# Overridable env vars (all optional):
#   GCS_COHORT_CSV_URI   Default: gs://su_vista_scratch/bikia_dev/lumia_cohort_progression_tasks-000000000000.csv
#   GCS_CORPUS_DIR_URI   Default: gs://vista_bench/thoracic_cohort_lumia
#   LOCAL_DATA_ROOT      Default: $HOME/data
#   LOCAL_OUTPUTS_ROOT   Default: $HOME/results
#   ENV_FILE             Default: <repo-root>/.env
#   MOUNT_ROOT           Default: $HOME/gcs     (used by --mount-gcs only)
#   GCS_BUCKETS          Default: "su_vista_scratch vista_bench" (--mount-gcs only)
#
set -euo pipefail

# ------------- defaults & flag parsing ----------------------------------------
MODE=none  # one of: none, copy, mount
for arg in "$@"; do
    case "$arg" in
        --copy-gcs)   MODE=copy ;;
        --mount-gcs)  MODE=mount ;;
        --help|-h)
            sed -n '3,29p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
    esac
done

GCS_COHORT_CSV_URI="${GCS_COHORT_CSV_URI:-gs://su_vista_scratch/bikia_dev/lumia_cohort_progression_tasks-000000000000.csv}"
GCS_CORPUS_DIR_URI="${GCS_CORPUS_DIR_URI:-gs://vista_bench/thoracic_cohort_lumia}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$HOME/data}"
LOCAL_OUTPUTS_ROOT="${LOCAL_OUTPUTS_ROOT:-$HOME/results}"
MOUNT_ROOT="${MOUNT_ROOT:-$HOME/gcs}"
GCS_BUCKETS="${GCS_BUCKETS:-su_vista_scratch vista_bench}"

# Repo root: this script lives at <repo>/scripts/setup_vm.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

log()  { printf '\033[1;34m[setup_vm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup_vm]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup_vm]\033[0m %s\n' "$*" >&2; exit 1; }

# ------------- 1. OS packages -------------------------------------------------
APT_PKGS=(build-essential git curl ca-certificates)

install_apt() {
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found — skipping OS package install (not Debian/Ubuntu?)"
        return
    fi

    # GCP VMs on default VPC have no IPv6 route, but apt resolves AAAA
    # records for deb.debian.org first and hangs/fails. Persist an IPv4-only
    # preference for apt so subsequent runs (and other users of the VM)
    # don't hit the same issue.
    local ipv4_conf=/etc/apt/apt.conf.d/99force-ipv4
    if [[ ! -f "$ipv4_conf" ]]; then
        log "pinning apt to IPv4 (creates $ipv4_conf)"
        echo 'Acquire::ForceIPv4 "true";' | sudo tee "$ipv4_conf" >/dev/null
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

# ------------- 2. uv ----------------------------------------------------------
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

# ------------- 3. PATH --------------------------------------------------------
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

# ------------- 4a. GCS copy ---------------------------------------------------
require_gsutil() {
    command -v gsutil >/dev/null 2>&1 \
        || die "gsutil not found. Install Google Cloud SDK or use --mount-gcs."
}

copy_gcs_data() {
    require_gsutil

    # CSV -> LOCAL_DATA_ROOT/<filename>
    local csv_name
    csv_name="$(basename "$GCS_COHORT_CSV_URI")"
    local local_csv="$LOCAL_DATA_ROOT/$csv_name"

    # XML corpus -> LOCAL_DATA_ROOT/<dirname>/
    local corpus_name
    corpus_name="$(basename "${GCS_CORPUS_DIR_URI%/}")"
    local local_corpus="$LOCAL_DATA_ROOT/$corpus_name"

    mkdir -p "$LOCAL_DATA_ROOT" "$LOCAL_OUTPUTS_ROOT"

    if [[ -f "$local_csv" ]]; then
        log "CSV already at $local_csv (skip download)"
    else
        log "downloading CSV: $GCS_COHORT_CSV_URI -> $local_csv"
        gsutil cp "$GCS_COHORT_CSV_URI" "$local_csv"
    fi

    mkdir -p "$local_corpus"
    log "syncing corpus: $GCS_CORPUS_DIR_URI -> $local_corpus (parallel; only new/changed)"
    # -m: parallel, -r: recursive. Only transfers files that differ; safe to rerun.
    gsutil -m rsync -r "${GCS_CORPUS_DIR_URI%/}" "$local_corpus"

    # Remember the resolved paths so write_env() can use them.
    _LOCAL_CSV="$local_csv"
    _LOCAL_CORPUS="$local_corpus"
    _LOCAL_OUTPUTS="$LOCAL_OUTPUTS_ROOT/fewshot_with_labels_outputs"
    mkdir -p "$_LOCAL_OUTPUTS"
}

# ------------- 4b. gcsfuse mount (alternative) --------------------------------
install_gcsfuse() {
    if command -v gcsfuse >/dev/null 2>&1; then
        log "gcsfuse already installed: $(gcsfuse --version 2>/dev/null | head -1)"
        return
    fi
    command -v apt-get >/dev/null 2>&1 || die "apt-get required for gcsfuse"
    log "installing gcsfuse"
    local codename
    codename="$(lsb_release -c -s 2>/dev/null || echo stable)"
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-${codename} main" \
      | sudo tee /etc/apt/sources.list.d/gcsfuse.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y fuse gcsfuse
}

mount_buckets() {
    mkdir -p "$MOUNT_ROOT"
    for bucket in $GCS_BUCKETS; do
        local mnt="$MOUNT_ROOT/$bucket"
        mkdir -p "$mnt"
        if mountpoint -q "$mnt"; then
            log "bucket '$bucket' already mounted at $mnt"
            continue
        fi
        log "mounting gs://$bucket at $mnt"
        gcsfuse --implicit-dirs -o allow_other "$bucket" "$mnt" \
            || warn "mount failed for $bucket (check storage.objectViewer auth)"
    done

    # Best-effort: derive the .env paths from the GCS URIs + mount root.
    _LOCAL_CSV="$(gsuri_to_mount_path "$GCS_COHORT_CSV_URI")"
    _LOCAL_CORPUS="$(gsuri_to_mount_path "$GCS_CORPUS_DIR_URI")"
    _LOCAL_OUTPUTS="$LOCAL_OUTPUTS_ROOT/fewshot_with_labels_outputs"
    mkdir -p "$LOCAL_OUTPUTS_ROOT" "$_LOCAL_OUTPUTS"
}

# gs://bucket/path/x  -> $MOUNT_ROOT/bucket/path/x
gsuri_to_mount_path() {
    local uri="$1"
    local rest="${uri#gs://}"
    echo "$MOUNT_ROOT/$rest"
}

# ------------- 5. .env generation ---------------------------------------------
write_env() {
    # If an existing .env contains literal placeholders from .env.example
    # (e.g. /home/USER/ or REPLACE_ME), it's clearly un-substituted — back it
    # up and regenerate. Preserve any VAULT_SECRET_KEY the user pasted in.
    local existing_key=""
    if [[ -f "$ENV_FILE" ]]; then
        local stale=0
        if grep -qE '/home/USER/|REPLACE_ME' "$ENV_FILE"; then
            stale=1
        fi
        existing_key="$(grep -E '^VAULT_SECRET_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
        if [[ "$stale" -eq 1 ]]; then
            local backup="${ENV_FILE}.bak.$(date +%s)"
            log "existing $ENV_FILE contains placeholders — backing up to $backup and regenerating"
            cp "$ENV_FILE" "$backup"
        else
            log ".env already exists at $ENV_FILE and looks substituted — not overwriting"
            log "  (delete it manually if you want the script to regenerate)"
            return
        fi
    fi

    log "writing $ENV_FILE"
    cat > "$ENV_FILE" <<EOF
# Generated by scripts/setup_vm.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
# Paste your VAULT_SECRET_KEY below and save — nothing else to edit.

VISTA_COHORT_CSV=$_LOCAL_CSV
VISTA_CORPUS_DIR=$_LOCAL_CORPUS
VISTA_OUTPUTS_DIR=$_LOCAL_OUTPUTS

# REQUIRED: paste your secure-llm key here.
VAULT_SECRET_KEY=${existing_key}
EOF
    chmod 600 "$ENV_FILE"
}

# ------------- 6. Verify ------------------------------------------------------
verify() {
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after install"
    log "uv version : $(uv --version)"
    log "git version: $(git --version)"
    if [[ "$MODE" != "none" ]]; then
        log "data paths:"
        log "  CSV    : $_LOCAL_CSV"
        log "  corpus : $_LOCAL_CORPUS"
        log "  outputs: $_LOCAL_OUTPUTS"
        log ".env    : $ENV_FILE"
    fi
    log ""
    log "next steps:"
    log "  1) paste VAULT_SECRET_KEY into $ENV_FILE"
    log "  2) source ~/.bashrc   (or open a new shell)"
    log "  3) cd $(printf '%q' "$REPO_ROOT") && uv sync"
    log "  4) follow experiments/fewshot_with_labels/README.md"
}

# ------------- run ------------------------------------------------------------
install_apt
install_uv
persist_path

case "$MODE" in
    copy)
        copy_gcs_data
        write_env
        ;;
    mount)
        install_gcsfuse
        mount_buckets
        write_env
        ;;
    none) ;;
esac

verify
