#!/bin/bash
set -euo pipefail

COMMAND="${1:-status}"
PAL_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PAL_PYTHON="$PAL_ROOT/.pal_venv/bin/python"
TEMPLATE="$PAL_ROOT/conf/launchd/com.pal.clink-worker.plist.template"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS/com.pal.clink-worker.plist"
LOG_DIR="$HOME/.pal/logs"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.pal.clink-worker"

worker_ready() {
  "$PAL_PYTHON" -c 'from utils.sqlite_conversation_storage import get_default_storage; s=get_default_storage(); expected={"codex":("gpt-5.6-sol","high"),"claude":("fable","xhigh")}; raise SystemExit(0 if all((c:=s.get_fresh_worker_capability(name,"default")) and (c["model"],c["reasoning_effort"])==policy for name,policy in expected.items()) else 1)'
}

if [ "$(uname -s)" != "Darwin" ]; then
  printf 'clink worker service currently supports macOS launchd only\n' >&2
  exit 2
fi

case "$COMMAND" in
  install)
    test -x "$PAL_PYTHON" || { printf 'missing PAL Python: %s\n' "$PAL_PYTHON" >&2; exit 1; }
    test -f "$TEMPLATE" || { printf 'missing template: %s\n' "$TEMPLATE" >&2; exit 1; }
    mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"
    chmod 700 "$LOG_DIR"
    PAL_PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    sed \
      -e "s|__PAL_PYTHON__|$PAL_PYTHON|g" \
      -e "s|__PAL_ROOT__|$PAL_ROOT|g" \
      -e "s|__PAL_PATH__|$PAL_PATH|g" \
      -e "s|__PAL_LOG_DIR__|$LOG_DIR|g" \
      "$TEMPLATE" > "$PLIST"
    chmod 600 "$PLIST"
    plutil -lint "$PLIST"
    launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    launchctl kickstart -k "$SERVICE"
    launchctl print "$SERVICE" >/dev/null
    ready=false
    for _attempt in $(seq 1 40); do
      if worker_ready; then
        ready=true
        break
      fi
      sleep 0.25
    done
    if [ "$ready" != true ]; then
      printf 'worker started but did not publish exact model capabilities\n' >&2
      exit 1
    fi
    printf 'installed and started %s\n' "$SERVICE"
    ;;
  status)
    launchctl print "$SERVICE"
    worker_ready
    ;;
  uninstall)
    launchctl bootout "$SERVICE" >/dev/null 2>&1 || true
    if [ -f "$PLIST" ]; then
      mv "$PLIST" "$PLIST.disabled"
      printf 'disabled plist retained at %s.disabled\n' "$PLIST"
    fi
    ;;
  *)
    printf 'usage: %s {install|status|uninstall}\n' "$0" >&2
    exit 2
    ;;
esac
