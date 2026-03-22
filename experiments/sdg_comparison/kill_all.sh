#!/usr/bin/env bash
# Kill all process groups launched by run_all.py.
#
# Usage:
#   bash experiments/sdg_comparison/kill_all.sh
#
# How it works:
#   run_all.py appends every subprocess's process-group ID to PID_FILE.
#   This script sends SIGTERM to each process group, then SIGKILL after 5s
#   if any are still alive.  Finally it clears the PID file.

PID_FILE="/tmp/sdg_comparison_pids.txt"
SIGNAL="${1:-TERM}"   # pass "KILL" as $1 to force-kill immediately

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE — nothing to kill."
    exit 0
fi

echo "Sending SIG${SIGNAL} to all registered process groups..."
while IFS= read -r pgid; do
    if [ -n "$pgid" ] && kill -0 -"$pgid" 2>/dev/null; then
        echo "  kill -${SIGNAL} -${pgid}"
        kill -"${SIGNAL}" -"$pgid" 2>/dev/null || true
    fi
done < "$PID_FILE"

if [ "$SIGNAL" = "TERM" ]; then
    echo "Waiting 5s for processes to exit gracefully..."
    sleep 5
    echo "Sending SIGKILL to any survivors..."
    while IFS= read -r pgid; do
        if [ -n "$pgid" ] && kill -0 -"$pgid" 2>/dev/null; then
            echo "  kill -KILL -${pgid}"
            kill -KILL -"$pgid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
fi

echo "Clearing PID file."
> "$PID_FILE"
echo "Done."
