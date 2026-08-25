#!/bin/bash
set -euo pipefail

COMMAND="${1:-status}"
PAL_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PAL_PYTHON="${PAL_PYTHON:-$PAL_ROOT/.pal_venv/bin/python}"
TEMPLATE="$PAL_ROOT/conf/launchd/com.pal.clink-worker.plist.template"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS/com.pal.clink-worker.plist"
LOG_DIR="$HOME/.pal/logs"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.pal.clink-worker"
RETRY_ATTEMPTS=40
RETRY_DELAY_SECONDS=0.25
LOCK_DIR="$HOME/.pal/clink-worker-service.lock"

release_lifecycle_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

acquire_lifecycle_lock() {
  local attempt
  for attempt in $(seq 1 "$RETRY_ATTEMPTS"); do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      trap release_lifecycle_lock EXIT HUP INT TERM
      return 0
    fi
    sleep "$RETRY_DELAY_SECONDS"
  done
  printf 'timed out waiting for PAL worker lifecycle lock: %s\n' "$LOCK_DIR" >&2
  return 1
}

service_loaded() {
  launchctl print "$SERVICE" >/dev/null 2>&1
}

wait_for_service_absent() {
  local attempt
  for attempt in $(seq 1 "$RETRY_ATTEMPTS"); do
    if ! service_loaded; then
      return 0
    fi
    sleep "$RETRY_DELAY_SECONDS"
  done
  printf 'service remained loaded after %s attempts: %s\n' \
    "$RETRY_ATTEMPTS" "$SERVICE" >&2
  return 1
}

bootstrap_worker() {
  local attempt bootstrap_error
  for attempt in $(seq 1 "$RETRY_ATTEMPTS"); do
    if bootstrap_error=$(launchctl bootstrap "$DOMAIN" "$PLIST" 2>&1); then
      return 0
    fi
    # launchd can retain a just-booted-out label briefly. Retry bootstrap only;
    # a visible old label is not proof that the new worker is ready.
    sleep "$RETRY_DELAY_SECONDS"
  done
  printf 'failed to bootstrap %s after %s attempts: %s\n' \
    "$SERVICE" "$RETRY_ATTEMPTS" "$bootstrap_error" >&2
  return 1
}

bootout_worker() {
  local bootout_error
  bootout_error=$(launchctl bootout "$SERVICE" 2>&1) || true
  # launchctl can return a nonzero status after it has already removed the
  # service. Require the observed absence before bootstrap so no new worker can
  # race a still-terminating old one.
  if wait_for_service_absent; then
    return 0
  fi
  printf 'failed to boot out still-loaded %s: %s\n' "$SERVICE" "$bootout_error" >&2
  return 1
}

worker_ready() {
  "$PAL_PYTHON" -m clink.readiness --quiet
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
    acquire_lifecycle_lock
    PAL_PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    sed \
      -e "s|__PAL_PYTHON__|$PAL_PYTHON|g" \
      -e "s|__PAL_ROOT__|$PAL_ROOT|g" \
      -e "s|__PAL_PATH__|$PAL_PATH|g" \
      -e "s|__PAL_LOG_DIR__|$LOG_DIR|g" \
      "$TEMPLATE" > "$PLIST"
    chmod 600 "$PLIST"
    plutil -lint "$PLIST"
    if service_loaded; then
      bootout_worker
    fi
    bootstrap_worker
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
    "$PAL_PYTHON" -m clink.readiness
    ;;
  uninstall)
    mkdir -p "$HOME/.pal"
    acquire_lifecycle_lock
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
