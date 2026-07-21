#!/usr/bin/env bash
# scripts/sweep_p1.sh
# ───────────────────
# Priority 1 timesteps sweep for the balanced/medium SAC profile.
# Hypothesis: the post-fix policy overfits past ~1M timesteps. We already
# have valid datapoints at 200k (Sharpe 0.83, Calmar 2.03) and 3M (Sharpe ~0,
# Calmar 0.64). This script fills in {300k, 600k, 1.2M, 2M} to find the
# generalization sweet spot, especially on Fold 2 (2024-H1 test window).
#
# Runs sequentially so the laptop isn't fighting itself. Each run writes its
# own log file under logs/sweep_p1/ and archives its trained model with a
# timesteps-tagged suffix. If the laptop dies mid-sweep, the completed runs
# are preserved; rerun the script and the prior tagged models won't be touched
# (only the active sweep step gets a fresh model file).

set -u   # treat unset vars as error; no `set -e` because a single failed
         # run shouldn't abort the rest of the sweep.

cd "$(dirname "$0")/.."

TIMESTEP_VALUES=(300000 600000 1200000 2000000)
LOG_DIR="logs/sweep_p1"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Priority 1 sweep — balanced/medium SAC"
echo "  Timestep values: ${TIMESTEP_VALUES[*]}"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

for steps in "${TIMESTEP_VALUES[@]}"; do
    tag="t${steps}"
    log_file="${LOG_DIR}/run_${tag}.log"

    # Skip if a tagged model already exists — assume that variant is done.
    # Lets us safely re-run the script after a laptop crash without redoing
    # completed work. Delete the tagged model file to force a rerun.
    if [ -f "models/sac_balanced_medium.${tag}.zip" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP ${tag} — already trained (models/sac_balanced_medium.${tag}.zip exists)"
        continue
    fi

    echo
    echo "────────────────────────────────────────────────────────────"
    echo "[$(date '+%H:%M:%S')] START ${tag} — log: ${log_file}"
    echo "────────────────────────────────────────────────────────────"

    # Clean state for this run. We do NOT touch any *.${tag}.zip archives.
    rm -f models/checkpoints/sac_balanced_medium_*_steps.zip
    rm -f models/sac_balanced_medium.zip models/sac_balanced_medium.vecnorm.pkl

    python3 -u backtest.py \
        --algo sac --risk balanced --term medium \
        --timesteps "${steps}" \
        > "${log_file}" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] FAIL ${tag} — exit code ${rc}. Check ${log_file}."
        continue
    fi

    if [ -f "models/sac_balanced_medium.zip" ]; then
        mv models/sac_balanced_medium.zip          "models/sac_balanced_medium.${tag}.zip"
        mv models/sac_balanced_medium.vecnorm.pkl  "models/sac_balanced_medium.${tag}.vecnorm.pkl"
        echo "[$(date '+%H:%M:%S')] DONE  ${tag} — model archived as models/sac_balanced_medium.${tag}.zip"
    else
        echo "[$(date '+%H:%M:%S')] WARN  ${tag} — no model file produced; skipping archive."
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Sweep complete: $(date)"
echo "════════════════════════════════════════════════════════════════"
