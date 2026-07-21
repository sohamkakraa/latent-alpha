#!/bin/bash
# schedule_night_training.sh
# ──────────────────────────
# Sets up a nightly cron job that runs the night_trainer at 11 PM every night.
# Run this once to install the schedule. It will auto-cancel in the morning.
#
# Usage: bash scripts/schedule_night_training.sh [--risk balanced] [--term medium]
#
# Remove the schedule with: crontab -l | grep -v night_trainer | crontab -

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$(which python3)"
LOG_FILE="$PROJECT_DIR/logs/cron.log"

RISK=${1:-balanced}
TERM=${2:-medium}

CRON_CMD="0 23 * * * cd $PROJECT_DIR && $PYTHON scripts/night_trainer.py --risk $RISK --term $TERM >> $LOG_FILE 2>&1"

# Add to crontab (avoid duplicates)
(crontab -l 2>/dev/null | grep -v "night_trainer"; echo "$CRON_CMD") | crontab -

echo "✓ Nightly training scheduled at 11 PM"
echo "  Risk: $RISK | Term: $TERM"
echo "  Logs: $LOG_FILE"
echo ""
echo "  To remove: crontab -l | grep -v night_trainer | crontab -"
echo "  To view:   crontab -l"
