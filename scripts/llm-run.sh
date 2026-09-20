#!/bin/bash
# Guarded launcher for llama.cpp experiments.
#   llm-run.sh [--max 48G] [--allow-existing] -- <command...>
# 1. Refuses to start while another llama-* loader is alive (the 2026-09-19 OOM came from two loaders).
# 2. Runs the command in its own systemd user scope with a hard memory cap, so a runaway
#    mmap/page-cache blow-up is reclaimed inside that cgroup instead of stalling tailscaled/sshd.
# 3. Marks the process as the preferred OOM victim.
set -u
MAX="${LLM_MAX_MEM:-48G}"; ALLOW=0
while [ $# -gt 0 ]; do case "$1" in
  --max) MAX="$2"; shift 2;; --allow-existing) ALLOW=1; shift;; --) shift; break;; *) break;; esac; done
[ $# -gt 0 ] || { echo "usage: $0 [--max 48G] [--allow-existing] -- <command...>" >&2; exit 2; }
if [ $ALLOW -eq 0 ]; then
  ALIVE=$(pgrep -fa 'llama-(server|perplexity|bench|cli|batched-bench)' | grep -v "$0" | cut -c1-100)
  if [ -n "$ALIVE" ]; then
    echo "REFUSING: another loader is alive (use --allow-existing to override):" >&2; echo "$ALIVE" >&2; exit 3
  fi
fi
AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo)
echo "[llm-run] MemAvailable=${AVAIL}G cap=${MAX} swap=off oom_score_adj=+800" >&2
exec systemd-run --user --scope --quiet \
  -p MemoryMax="$MAX" -p MemorySwapMax=0 \
  --unit "llm-$(date +%H%M%S)-$$" -- choom -n 800 -- "$@"
