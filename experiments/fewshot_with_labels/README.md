# fewshot_with_labels experiment

Tests whether showing the LLM **similar patients' vignettes + their ground-truth Yes/No answers for the same task** improves binary predictions over a no-retrieval baseline. Retrieval is held fixed; only the level of context detail varies, selected at run time by `--context`.

## Design

### Data model (normalized)

Two JSONL-backed shapes. `PatientState` is keyed by `(person_id, embed_time)` — a patient may have several states when different tasks use different prediction times. `LabeledItem` carries its own `embed_time` and resolves to the corresponding state via that composite key.

- **`PatientState`** (one per unique `(person_id, embed_time)`): `split`, precomputed `vignette` built from the XML timeline chopped at `embed_time`, `created_at`.
- **`LabeledItem`** (one per `(person_id, task)` after dropping `label == -1`): `task`, `task_group`, `question`, `label`, `label_description`, `split`, `embed_time`, `created_at`.
- **`CohortItem`** (join view): `store.join(pid, task)` returns the state at `(pid, item.embed_time)` joined with the item.

Timelines are NOT stored — they are regenerated from the XML corpus at prompt-build time via `DeterministicTimelineLinearizationGenerator`. Precompute deduplicates by `(pid, embed_time)`, so two tasks sharing a prediction time share one vignette (no redundant LLM calls).

### Retrieval (fixed)

**BM25 similarity is computed on vignettes only, never on raw timelines** — regardless of `--context`. One BM25 index per task:

- Candidates: patients in the `train` split with `label != -1` for that task.
- Each candidate's document is their **task-aligned vignette** — the `PatientState` at `(candidate.person_id, candidate_item.embed_time)`. When a patient has different embed_times across tasks, the vignette used for BM25 (and for the LLM context) is the one built at the embed_time matching the task being searched.
- The query's retrieval vignette is likewise the state at `(query.person_id, query_item.embed_time)`.

Implemented in `src/meds_mcp/similarity/task_retriever.py` (`TaskAwareRetriever`).

### LLM context (variable, one per run)

| `--context` | Query representation | Similars |
|---|---|---|
| `baseline_vignette` | Vignette | *(none)* |
| `vignette` | Vignette | Similar vignettes + Yes/No label for this task |
| `baseline_timeline` | Timeline | *(none)* |
| `timeline` | Timeline | Similar timelines (chopped at each similar's own `embed_time`) + Yes/No label |

The four variants form two paired 1:1 comparisons where only the similars block differs:

- `baseline_vignette` ↔ `vignette` — effect of adding similars, query rendered as vignette in both.
- `baseline_timeline` ↔ `timeline` — effect of adding similars, query rendered as timeline in both.

`analyze_results.py` uses these pairings when computing flip / fix / hurt rates. Cross-pair comparisons (`baseline_vignette` vs `baseline_timeline`, `vignette` vs `timeline`) isolate the effect of query representation alone.

Similar patients' question text is omitted (same task ⇒ identical wording, just wastes tokens). Label is shown as `Yes` / `No` (from `label_description`), not the numeric code.

### Evaluation defaults

- Query pool: sampled from a split (`valid` by default, `test` optional); retrieval candidates come only from `train`.
- LLM: `temperature=0`, seed logged per row. Output parsed to `Yes`/`No` and compared to ground truth.
- Per-task n may be small depending on pool size; `analyze_results.py` flags tasks with `n < 10`. **Primary metric: overall accuracy.**

---

## Setup

### Local

```bash
uv sync                               # installs Python deps from pyproject.toml + uv.lock
cp .env.example .env                  # then edit .env with your VAULT_SECRET_KEY
```

`.env` is auto-loaded on import by `experiments/fewshot_with_labels/_paths.py`. On a laptop with the repo's default data layout, no env vars need to be set.

### GCP VM (or any Debian/Ubuntu host)

The scripts use plain `pathlib`/`open()`, so `gs://...` URIs don't work directly. The setup script copies data from GCS to local disk and writes a `.env` pointing at the local copies — you only paste the vault key afterward.

#### Creating the VM

Create the VM on Stanford's allowlisted `shc-som-secure-gpt-<region>` subnet. That subnet handles egress to `apim.stanfordhealthcare.org` upstream, so the VM can safely use `--no-address` (no external IP) and still reach APIM, GitHub, and GCS.

```bash
# --- variables -------------------------------------------------------------
export GOOGLE_CLOUD_PROJECT=som-nero-plevriti-deidbdf
export GOOGLE_CLOUD_LOCATION=us-west1-a

# SHC_GCP_NETWORK strips the trailing zone letter (-a) so that us-west1-a
# maps to subnet `shc-som-secure-gpt-us-west1` (no "-a" suffix).
export SHC_GCP_NETWORK=$(echo shc-som-secure-gpt-$GOOGLE_CLOUD_LOCATION | sed 's/-[^-]*$//')

export GOOGLE_HOST=bikia-pinnacle-e2   # name of your new VM

# --- create ----------------------------------------------------------------
gcloud compute instances create "$GOOGLE_HOST" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --zone="$GOOGLE_CLOUD_LOCATION" \
  --machine-type=e2-standard-2 \
  --network=default \
  --subnet="$SHC_GCP_NETWORK" \
  --no-address \
  --image=debian-12-bookworm-v20260114 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --tags=iap-ssh
```

Key flags and why:
- `--subnet=$SHC_GCP_NETWORK` — Stanford's `shc-som-secure-gpt-us-west1` subnet is allowlisted on APIM. Without it, LLM calls time out (`ConnectTimeout to apim.stanfordhealthcare.org`).
- `--no-address` — required on this subnet; egress routing is handled upstream by Stanford, not via a VM-level external IP.
- `--machine-type=e2-standard-2` — 2 vCPU / 8 GB RAM; leaves headroom for 40 BM25 indices + XML parsing. `e2-medium` (4 GB) OOMs intermittently; `e2-standard-4` is comfortable once you parallelize LLM calls.
- `--boot-disk-size=50GB` — corpus (~7.5 GB) + uv env + deps + outputs fits; 10 GB is **not** enough. You'll see a harmless "Disk size larger than image" warning on create; Debian 12 auto-resizes the root partition on first boot.
- `--scopes=cloud-platform` — lets `gsutil` (run by `scripts/setup_vm.sh --copy-gcs`) read the source buckets without an interactive `gcloud auth` flow.
- `--tags=iap-ssh` — lets you connect with `gcloud compute ssh <name> --tunnel-through-iap` without a public SSH port.

Verify the VM landed on the right subnet with no external IP:

```bash
gcloud compute instances describe "$GOOGLE_HOST" \
  --project="$GOOGLE_CLOUD_PROJECT" --zone="$GOOGLE_CLOUD_LOCATION" \
  --format="get(networkInterfaces[0].subnetwork,networkInterfaces[0].accessConfigs)"
# Expected: subnetwork URI ends in /shc-som-secure-gpt-us-west1
#           accessConfigs is empty ([])
```

SSH in via IAP tunnel:

```bash
gcloud compute ssh "$GOOGLE_HOST" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --zone="$GOOGLE_CLOUD_LOCATION" \
  --tunnel-through-iap
```

Once you get a shell prompt on the VM, sanity-check that the subnet's egress is working **before** running anything LLM-dependent:

```bash
# APIM reachable? A 404 here is success — it means the hostname resolves and
# TLS handshake works; only the root path doesn't exist as an endpoint.
curl -4 -I --connect-timeout 10 https://apim.stanfordhealthcare.org/

# GCS + GitHub (needed for setup_vm.sh --copy-gcs)
curl -4 -I --connect-timeout 10 https://storage.googleapis.com/
curl -4 -I --connect-timeout 10 https://github.com/
```

All three must return HTTP headers (any 2xx/3xx/4xx is fine; timeouts are not). If APIM times out, contact the secure-gpt team — this VM's subnet egress or service account may need allowlisting.

#### Why `--no-address` here (vs a VM with an external IP)

On **this specific Stanford subnet**, outbound traffic is NAT'd through Stanford's infrastructure and Stanford's allowlisted egress IP reaches APIM. A VM with its own external IP (no SHC subnet) would use a random GCP egress IP that is **not** on APIM's allowlist, so LLM calls would fail with `ConnectTimeout`. Use the subnet; skip the external IP.

#### First-time bootstrap on the VM

Debian 12 minimal doesn't ship `git`, so the very first step is a one-line apt install. After that, `scripts/setup_vm.sh` handles everything else (including git on future runs). Note the `-o Acquire::ForceIPv4=true` — GCP default VPCs have no IPv6 route, and apt otherwise tries AAAA records first and fails with `Network is unreachable`.

```bash
# 1) Install git (one-line; only needed the very first time).
#    ForceIPv4 works around GCP's no-IPv6-route default.
sudo apt-get -o Acquire::ForceIPv4=true update
sudo apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends git

# 2) Clone over HTTPS (private repos: use a GitHub personal access token
#    when prompted). SSH also works if you've added a VM key to GitHub —
#    see the SSH note further down.
git clone --single-branch --branch pinnacle \
  https://github.com/VISTA-Stanford/meds-mcp.git
cd meds-mcp

# 3) Installs apt + uv, copies CSV and XML corpus from GCS to local disk,
#    generates .env with the correct local paths. One command.
bash scripts/setup_vm.sh --copy-gcs
source ~/.bashrc

# 4) Paste VAULT_SECRET_KEY into the generated .env — only manual step.
${EDITOR:-nano} .env

# 5) Install Python deps
uv sync

# 6) Sanity-check outbound reachability BEFORE the long precompute
curl -I --connect-timeout 10 https://apim.stanfordhealthcare.org/
curl -I --connect-timeout 5 https://storage.googleapis.com/
```

If either `curl` times out, fix networking before running any LLM-dependent step (see "Troubleshooting" further down).

##### Skip the manual git install on new VMs

Bake git installation into the VM's startup script so your very first SSH has it:

```bash
# Add this to your create command:
--metadata=startup-script='#!/bin/bash
apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl'
```

##### SSH clone (optional, for push access)

HTTPS is fine for read-only clones. If you want to push from the VM, register an SSH key with GitHub:

```bash
ssh-keygen -t ed25519 -C "$GOOGLE_HOST" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# Paste into https://github.com/settings/keys
ssh -T git@github.com   # confirm
```

**If `ssh -T git@github.com` hangs or says "Network is unreachable":** the SHC subnet likely blocks outbound TCP/22 (only 443 egress is allowed). Use GitHub's SSH-over-443 endpoint:

```bash
cat >> ~/.ssh/config <<'EOF'

Host github.com
  HostName ssh.github.com
  Port 443
  User git
EOF
chmod 600 ~/.ssh/config
ssh -T git@github.com   # should now say "Hi <your-username>!…"
```

After that, `git@github.com:...` URLs route over 443 transparently and work from the SHC subnet.

**Auth:** `gsutil` (and the fallback gcsfuse) need read access to the source buckets. Either attach a service account with `storage.objectViewer` to the VM, or run `gcloud auth application-default login` first. Create the VM with `--scopes=cloud-platform` for the zero-config path.

**Defaults used by `--copy-gcs`:**

| | Source | Local destination |
|---|---|---|
| CSV | `gs://su_vista_scratch/bikia_dev/lumia_cohort_progression_tasks-000000000000.csv` | `$HOME/data/lumia_cohort_progression_tasks-000000000000.csv` |
| XML corpus | `gs://vista_bench/thoracic_cohort_lumia/` | `$HOME/data/thoracic_cohort_lumia/` |
| Outputs | — | `$HOME/results/fewshot_with_labels_outputs/` |

Override any of them at call time:

```bash
GCS_COHORT_CSV_URI=gs://my-bucket/some.csv \
GCS_CORPUS_DIR_URI=gs://my-bucket/corpus \
LOCAL_DATA_ROOT=$HOME/scratch \
  bash scripts/setup_vm.sh --copy-gcs
```

Rerunning the script is safe: the CSV download skips if already present, and `gsutil -m rsync -r` for the corpus only transfers new/changed files.

**Alternative: `--mount-gcs`** — if you prefer gcsfuse mounts over copying (useful when the corpus is too big to host locally). Same one-command flow, generates a `.env` pointing at `$HOME/gcs/<bucket>/...`. Noticeably slower for the 8k-XML precompute step than local disk, though.

### Path resolution precedence (highest → lowest)

1. CLI flag (`--csv`, `--corpus-dir`, `--output-dir`, `--patients`, `--items`, `--pool`).
2. Env var already exported in the shell.
3. `.env` at repo root (auto-loaded).
4. Built-in local defaults (`data/collections/vista_bench/...`, `experiments/fewshot_with_labels/outputs/`).

| Env var | What it points at |
|---|---|
| `VISTA_COHORT_CSV` | Source CSV for `build_cohort.py` |
| `VISTA_CORPUS_DIR` | Directory of per-patient `{pid}.xml` files |
| `VISTA_OUTPUTS_DIR` | Where all experiment artifacts are written |
| `VAULT_SECRET_KEY` | Secure-llm auth (read by `securellm.Client` directly) |

---

## Run — exact commands (from repo root)

### 1. Build cohort (fast, no LLM)

```bash
uv run python experiments/fewshot_with_labels/build_cohort.py
# -> outputs/patients.jsonl  (one PatientState per (pid, embed_time))
# -> outputs/items.jsonl     (one LabeledItem per (pid, task), label != -1)
```

### 2. Precompute vignettes (~8k LLM calls; resumable)

Wall-clock is 2–7 hours depending on rate-limiting. Run it under `tmux` so a dropped SSH session doesn't kill the process.

```bash
# Install tmux if you're on a fresh VM (IPv4 forced — GCP default VPC has no IPv6 route).
sudo apt-get -o Acquire::ForceIPv4=true install -y tmux

# Start a named session and cd into the repo.
tmux new -s precompute
cd ~/meds-mcp

# Load .env so VAULT_SECRET_KEY is in this shell's environment.
# (Experiment scripts auto-load .env on import, but exporting it here
# makes ad-hoc Python one-liners work too.)
set -a && source .env && set +a

# Kick off the precompute. `tee` mirrors output to a log file for grepping later.
mkdir -p logs
uv run python experiments/fewshot_with_labels/precompute_vignettes.py \
  --model apim:gpt-4.1-mini 2>&1 | tee logs/precompute_$(date +%s).log
```

Default behaviour: all events on/before `embed_time` (`--n-encounters 0`), USMLE-style prompt. Rerun the same command to resume after a crash — it skips states that already have a vignette. Add `--force` to regenerate all vignettes (e.g. after changing `configs/prompts/vignette_prompt.txt`). Use `--limit 30` for a small smoke test before the full run.

**Useful tmux keys:**

| Action | Keys |
|---|---|
| Detach and leave job running | `Ctrl-b` then `d` |
| Reattach (after SSH reconnect) | `tmux attach -t precompute` |
| Scroll back through output | `Ctrl-b [`, arrow keys / PgUp, `q` to exit |
| List sessions | `tmux ls` |
| Split window vertically / horizontally | `Ctrl-b "` / `Ctrl-b %` |
| Switch between panes | `Ctrl-b o` |
| Kill the session | `tmux kill-session -t precompute` |

**Monitor progress from a split pane:**

```bash
# After Ctrl-b " in tmux:
watch -n 5 'free -h; df -h / | tail -1; echo; \
  wc -l $HOME/results/fewshot_with_labels_outputs/patients.jsonl'
```

**Grep the log for skip reasons once finished:**

```bash
grep -E "Timeline extract failed|Empty timeline|Vignette LLM failed" logs/precompute_*.log
```

The same `tmux new -s <name>` pattern works for steps 4 and 5 (the LLM-bound runs); just pick a different session name per run.

### 3. Sample the evaluation pool

Pick one pattern depending on what you want to evaluate on:

```bash
# A) Pilot: 100 random pids from valid (seeded, reproducible).
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --split valid --n 100 --seed 42 --require-vignette --require-item
# -> outputs/pool_valid_100.json

# B) Full test-split run: every eligible test patient (no sampling).
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --split test --all --require-vignette --require-item
# -> outputs/pool_test_all.json
```

`--require-vignette` drops patients whose state has no vignette (e.g. `embed_time` before their first XML event). `--require-item` drops patients with zero non-(−1) items.

### 4. Run the experiment — one invocation per `--context`

```bash
# Change --pool and --query-split to match the pool you sampled in step 3.
POOL=experiments/fewshot_with_labels/outputs/pool_test_all.json
QSPLIT=test

# Baselines: no similars in the prompt → --top-k is not used.
uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context baseline_vignette --pool $POOL --query-split $QSPLIT \
  --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context baseline_timeline --pool $POOL --query-split $QSPLIT \
  --model apim:gpt-4.1-mini

# With-similars variants: --top-k controls how many exemplars are injected.
uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context vignette --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context timeline --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini
```

The scripts skip the BM25 build entirely for `baseline_*` variants (no similars shown → no retrieval). If you do pass `--top-k` on a baseline context, the script logs a warning and ignores it; `top_k` is recorded as `null` in the result rows so the metadata honestly reflects "no retrieval happened". Each `--context` writes its own `experiment_results_<context>.jsonl` and can be rerun independently.

### 5. Aggregate + compare to ground truth

```bash
uv run python experiments/fewshot_with_labels/analyze_results.py
# -> outputs/analysis_summary.json    per-variant accuracy, per-task matrix, paired flip/fix/hurt
# -> outputs/per_task_accuracy.csv    wide table for spreadsheet import
# also prints a human-readable per-task table and paired-comparison block to stdout
```

---

## Inspection utilities

### Preview the prompt (no LLM call)

```bash
# Pick first pool pid + first non-(-1) task, print all 4 context prompts:
uv run python experiments/fewshot_with_labels/show_prompt.py --context all

# Specific (pid, task), single context:
uv run python experiments/fewshot_with_labels/show_prompt.py \
  --person-id 135908791 --task died_any_cause_1_yr --context vignette
```

### Sanity-check a results file

```bash
# Per-context row count and parse rate:
for ctx in baseline_vignette vignette baseline_timeline timeline; do
  f=experiments/fewshot_with_labels/outputs/experiment_results_$ctx.jsonl
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  p=$(grep -c '"pred": "\(Yes\|No\)"' "$f")
  echo "$ctx: $p/$n parsed"
done
```

---

## Outputs

Default: `experiments/fewshot_with_labels/outputs/` (or `$VISTA_OUTPUTS_DIR`). Directory is gitignored.

| File | Step | Description |
|---|---|---|
| `patients.jsonl` | 1, 2 | `PatientState` per `(person_id, embed_time)`; vignette filled by step 2 |
| `items.jsonl` | 1 | `LabeledItem` per `(person_id, task)` after dropping `label=-1` |
| `pool_<split>_<tag>.json` | 3 | Sampled evaluation pool (sorted list of person_ids) |
| `pool_<split>_<tag>_manifest.json` | 3 | Sampling config (split, seed, sizes, whether `--all` was used) |
| `experiment_results_<context>.jsonl` | 4 | One row per (pid, task): context, pred, raw, label, similars, model/seed/temperature, per-row prompt token counts (`prompt_tokens_system` / `_user` / `_total`) |
| `experiment_meta_<context>.json` | 4 | Run settings + aggregate prompt-token stats under `prompt_token_stats` (`user_tokens` and `total_tokens` each include `n / min / max / mean / median / p10 / p90 / total`) |
| `analysis_summary.json` | 5 | Per-variant accuracy, per-task matrix, paired flip/fix/hurt |
| `per_task_accuracy.csv` | 5 | Flat per-task × per-variant table for spreadsheets |

---

## CLI reference

### `build_cohort.py`
- `--csv` (default: `VISTA_COHORT_CSV` → local CSV) — source of rows.
- `--output-dir` (default: `VISTA_OUTPUTS_DIR`) — where to write patients/items.
- `--drop-label` — labels to drop (repeatable; default `-1`).

### `precompute_vignettes.py`
- `--patients` / `--items` — cohort JSONLs (defaults honor env vars).
- `--corpus-dir` (default: `VISTA_CORPUS_DIR`) — XML directory.
- `--n-encounters` (default: **0** = all events on/before `embed_time`).
- `--max-input-tokens` (default: **unset → effective input budget ≈ 116928** for `apim:gpt-4.1-mini`) — cap the linearized timeline at this many tokens (`cl100k_base`) before sending it to the summarizer. Oversized timelines are head+tail-truncated with an explicit `[truncated]` marker. Set to `0` to disable truncation. By default you get as close to the model's advertised 128K context as APIM deployments practically allow (128K − 1024 output − 2048 safety margin); in practice virtually every patient in the cohort fits without truncation. Pass a smaller integer (e.g. 32768) only if you want faster / cheaper runs at the cost of truncating longer trajectories.
- `--model-context-tokens` (default: **120000**) — advertised model context window. Used (a) to validate `--max-input-tokens` at startup (the script errors out if you pass a value > the effective input budget) and (b) logged in the run meta. `120000` is the observed-safe value for `apim:gpt-4.1-mini`; nominal is 128K but APIM deployments typically reserve ~2–4K for wrapper overhead.
- `--max-output-tokens` (default: **1024**) — max tokens the summarizer may generate. Subtracted from the context window to derive the effective input budget. Keep aligned with the summarizer's `generation_overrides.max_tokens`.
- `--token-safety-margin` (default: **2048**) — extra buffer reserved on top of `--max-output-tokens` (covers the system prompt + `cl100k_base`-vs-`o200k_base` tokenizer drift).
- `--max-retries` (default: **3**) — per-patient retries on LLM errors (transient APIM timeouts, rate limits). Total attempts = `max_retries + 1`. Exponential backoff.
- `--retry-backoff-seconds` (default: **2**) — base sleep between retries; doubles each attempt.
- `--model` (default: `apim:gpt-4.1-mini`).
- `--limit N` — process only N states (smoke).
- `--force` — regenerate even if vignette already present.
- `--no-progress` — disable tqdm.

On success, each written `PatientState` now records a `vignette_input_was_truncated: bool` flag so you can target only truncated patients for re-analysis later without re-summarizing the full cohort.

### `sample_pool.py`
- `--split` (default `valid`).
- `--n N` (default 100) — or `--all` to include every eligible pid.
- `--seed` (default 42; ignored under `--all`).
- `--require-vignette` / `--require-item` — skip empty-vignette or no-item patients.

### `run_experiment.py`
- `--context {baseline_vignette,baseline_timeline,vignette,timeline}` *(required)*.
- `--pool` — JSON list of pids (produced by step 3).
- `--query-split` (default `valid`) / `--candidate-split` (default `train`).
- `--top-k` (default 3) — similar patients injected into the prompt. **Ignored for `baseline_*` contexts** (a warning is logged if you pass it anyway, and `top_k` is recorded as `null` in the output).
- `--n` — process only first N pids in the pool (debug).
- `--n-encounters` (default 0) — capacity on timeline context blocks.
- `--max-chars` (default 120_000) — per-block timeline truncation (character-level; belt-and-suspenders alongside the token-level cap).
- `--max-prompt-tokens` (default: **auto** from model context) — token-level cap on the rendered user prompt (plus system prompt). When the candidate prompt exceeds this, progressive trim runs: (1) drop the largest neighbor block, repeat until under budget; (2) if still over, head+tail-truncate the query block. Pass `0` to disable. Auto-default is `effective_input_budget(model_context, max_output, safety_margin, system_tokens)` ≈ 116900 for `apim:gpt-4.1-mini`.
- `--model-context-tokens` (default **120000**), `--max-output-tokens` (default **8**), `--token-safety-margin` (default **2048**) — same semantics as precompute; used to compute the auto `--max-prompt-tokens`. `max-output-tokens` is 8 here (Yes/No only) rather than 1024 (full vignette).
- `--llm-retries` (default **1**) — one-shot retry on LLM exceptions. No input mutation. Protects against transient APIM blips without inflating `n_skipped_rows`.
- `--model`, `--seed`, `--temperature`, `--delay-seconds`.

Every result row now records `prompt_tokens_before_trim`, `neighbors_dropped_count`, `neighbors_dropped_ids`, and `query_truncated`. The run meta file adds a `prompt_trim_stats` block aggregating those counts across rows.

### `analyze_results.py`
- `--input-dir` / `--output-dir` (defaults: `VISTA_OUTPUTS_DIR`).

### `show_prompt.py`
- `--person-id`, `--task`, `--context` (or `all`), `--top-k`, `--show-system`.

---

## Defensive invariants

- `TaskAwareRetriever.retrieve` asserts every returned neighbor has `split == "train"` (or whatever `--candidate-split` was set to) and `label not in drop_labels`.
- BM25 is invariantly vignette↔vignette — comment at `run_experiment.py` call site and `task_retriever.py` module docstring both note this.
- `CohortStore` rejects duplicate `(person_id, embed_time)` keys when constructed.

---

## Troubleshooting

### `ConnectTimeout` / `Max retries exceeded` to `apim.stanfordhealthcare.org`

The VM can't reach Stanford's secure-llm gateway. Usually one of:

1. **VM created with `--no-address` and no Cloud NAT** — VM has no outbound internet at all. Either recreate without `--no-address`, or add Cloud NAT on the VPC.
2. **VM's external IP isn't on Stanford's APIM allowlist** — get the IP (`curl -s https://api.ipify.org`) and ask the secure-llm team to allowlist it. Consider reserving a static external IP first so it doesn't change on reboot.
3. **DNS misconfigured** — rare on GCP; test with `getent hosts apim.stanfordhealthcare.org`.

Before kicking off any LLM-dependent step, verify with a one-line smoke test:

```bash
uv run python -c "
from meds_mcp.server.llm import get_llm_client, get_default_generation_config
c = get_llm_client('apim:gpt-4.1-mini')
r = c.chat.completions.create(model='apim:gpt-4.1-mini',
    messages=[{'role':'user','content':'say ok'}],
    **get_default_generation_config({'temperature':0,'max_tokens':4}))
print(r.choices[0].message.content)
"
```

### `.env` contains literal `/home/USER/` placeholders

You probably ran `cp .env.example .env` before `bash scripts/setup_vm.sh --copy-gcs`. The setup script now detects this and auto-regenerates (backing up to `.env.bak.<timestamp>` and preserving any `VAULT_SECRET_KEY` you pasted). Just rerun:

```bash
bash scripts/setup_vm.sh --copy-gcs
```

### `build_cohort.py` fails with `PermissionError: '/home/USER'`

Same root cause as the placeholder issue above — `.env` points at `/home/USER/...` which doesn't exist. Fix with the rerun above.

### `No ... patients with vignette` on the test pool

Your `sample_pool.py` ran before `precompute_vignettes.py` finished. Rerun `sample_pool.py --split test --all --require-vignette --require-item` after precompute completes.

### 47 states skipped as `empty_timeline`

Expected — patients whose `embed_time` precedes their first XML event. Nothing to summarize. These are excluded from retrieval and from the eval pool automatically (via `--require-vignette`). Run `scripts/diag_embed_before_xml.py` for an exact count of these on your cohort.

### Many `Vignette LLM failed: Expecting value: line 1 column 1 (char 0)`

That's APIM returning an empty response body — the secure-llm client then fails parsing it as JSON. This is prevented proactively by the new token-budget flags:

- `precompute_vignettes.py` validates `--max-input-tokens` against the effective input budget derived from `--model-context-tokens` / `--max-output-tokens` / `--token-safety-margin` at startup and errors out if it would exceed the model's context. The default `--max-input-tokens=65536` leaves ample headroom on `apim:gpt-4.1-mini`.
- `run_experiment.py` applies an auto `--max-prompt-tokens` (also derived from the same budget) and progressively trims when a specific prompt exceeds it (drops the largest neighbor first; finally head+tail-truncates the query block).

If you still see this error, either (a) you explicitly lowered `--max-prompt-tokens` / raised input past the model's actual limit, or (b) the secure-llm connection itself is failing. Verify APIM reachability (`curl -I https://apim.stanfordhealthcare.org/`) and re-run. The existing retry-with-backoff in `precompute_vignettes.py` and the one-shot retry in `run_experiment.py` recover from transient blips automatically.

### `apt-get install` fails with `Network is unreachable` / IPv6 errors

The GCP default VPC has no IPv6 route, but apt tries AAAA records first. Force IPv4, either once:

```bash
sudo apt-get -o Acquire::ForceIPv4=true update
sudo apt-get -o Acquire::ForceIPv4=true install -y <pkg>
```

or permanently (what `scripts/setup_vm.sh` does automatically):

```bash
echo 'Acquire::ForceIPv4 "true";' | sudo tee /etc/apt/apt.conf.d/99force-ipv4
```

If you still see `Network is unreachable` on IPv4 after that, the VM has no outbound internet at all — see the APIM/networking notes above (Cloud NAT, drop `--no-address`).
