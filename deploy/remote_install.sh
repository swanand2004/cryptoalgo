#!/usr/bin/env bash
# Runs ON the VPS, as root, invoked by .github/workflows/deploy.yml after it has
# uploaded it with /tmp/deploy_app.tgz, /tmp/deploy_dist.tgz, /tmp/requirements.txt.
#
# Same steps as deploy_vps.sh (back up running code, unpack, chown, restart),
# plus two things the manual script left to a human reading the screen:
#   1. requirements.txt changes are installed, with the old package set saved;
#   2. the deploy is only "done" when the TRADING LOOP is alive on the new code,
#      not merely when systemd says active. If it is not, the previous code and
#      packages are restored, the service restarted, and this exits non-zero so
#      the workflow goes red.
set -euo pipefail

ROOT=/opt/cryptoalgo
SERVICE=cryptoalgo
PY=$ROOT/backend/.venv/bin/python
TS=$(date -u +%Y%m%d_%H%M%S)
BACKUP=$ROOT/backups/code_$TS.tgz
FREEZE=""

# A loop that is alive writes a paper equity snapshot at start-up and every 60 s
# (scheduler._record_equity). Pass = the newest snapshot is younger than the
# restart AND younger than FRESH_S, checked after MIN_UP_S so at least one
# interval tick has run, not just the start-up one.
MIN_UP_S=150
FRESH_S=90
GIVE_UP_S=420

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

cd "$ROOT"
mkdir -p backups

trading_age() {
  # Prints age_seconds from /api/health/trading, or "none" if unreachable/empty.
  "$PY" - <<'EOF' 2>/dev/null || echo none
import json, urllib.request, urllib.error
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/api/health/trading", timeout=5)
    body = r.read()
except urllib.error.HTTPError as e:  # 503 = stale, body still carries the age
    body = e.read()
age = json.loads(body).get("age_seconds")
print("none" if age is None else int(age))
EOF
}

wait_healthy() {
  local start=$SECONDS up age
  while :; do
    sleep 10
    up=$((SECONDS - start))
    if ! systemctl is-active --quiet "$SERVICE"; then
      log "   service not active (${up}s)"
    else
      age=$(trading_age)
      log "   up ${up}s, last equity snapshot age: ${age}s"
      if [[ "$age" != none && $up -ge $MIN_UP_S && $age -lt $FRESH_S && $age -lt $up ]]; then
        return 0
      fi
    fi
    [[ $up -ge $GIVE_UP_S ]] && return 1
  done
}

rollback() {
  trap - ERR
  set +e
  log "Rolling back to $BACKUP"
  journalctl -u "$SERVICE" -n 60 --no-pager
  rm -rf backend/app frontend/dist
  tar xzf "$BACKUP" --no-same-owner
  chown -R cryptoalgo:cryptoalgo backend/app frontend/dist
  if [[ -n "$FREEZE" ]]; then
    log "Restoring previous Python packages from $FREEZE"
    cp "backups/requirements_$TS.txt" backend/requirements.txt 2>/dev/null
    sudo -u cryptoalgo "$PY" -m pip install -q -r "$FREEZE"
  fi
  systemctl restart "$SERVICE"
  if wait_healthy; then
    log "ROLLBACK OK: previous code is running and trading"
  else
    log "ROLLBACK ALSO UNHEALTHY -- check the server by hand now"
  fi
  exit 1
}

# The nightly retrain (cryptoalgo-retrain.service, 03:00 UTC, up to 3 h) runs
# from this same backend/app. Swapping the code under it can crash it mid-run
# or train a model on half old, half new code. Nothing has been touched yet, so
# just stop; re-run the deploy once it finishes.
if systemctl is-active --quiet cryptoalgo-retrain.service; then
  log "ABORTED: nightly retrain is running; nothing was changed. Re-run the deploy when it finishes."
  exit 1
fi

log "1/5 Backing up running code -> $BACKUP"
tar czf "$BACKUP" backend/app frontend/dist
# From here on, any failing command restores the backup instead of leaving
# half-installed code or packages behind.
trap rollback ERR

if ! cmp -s /tmp/requirements.txt backend/requirements.txt; then
  FREEZE=$ROOT/backups/pip_freeze_$TS.txt
  log "2/5 requirements.txt changed: saving package set -> $FREEZE, installing"
  sudo -u cryptoalgo "$PY" -m pip freeze > "$FREEZE"
  cp backend/requirements.txt "backups/requirements_$TS.txt" 2>/dev/null || true
  install -o cryptoalgo -g cryptoalgo -m 644 /tmp/requirements.txt backend/requirements.txt
  sudo -u cryptoalgo "$PY" -m pip install -q -r backend/requirements.txt
else
  log "2/5 requirements.txt unchanged"
fi

log "3/5 Installing new code"
tar xzf /tmp/deploy_app.tgz -C backend --no-same-owner
rm -rf frontend/dist && tar xzf /tmp/deploy_dist.tgz -C frontend --no-same-owner
chown -R cryptoalgo:cryptoalgo backend/app frontend/dist

log "4/5 Restarting $SERVICE"
systemctl restart "$SERVICE"

log "5/5 Waiting for the trading loop to prove it is alive (up to ${GIVE_UP_S}s)"
if wait_healthy; then
  rm -f /tmp/deploy_app.tgz /tmp/deploy_dist.tgz /tmp/requirements.txt /tmp/remote_install.sh
  trap - ERR
  log "DEPLOY OK: trading loop alive on the new code"
  exit 0
fi

trap - ERR
log "DEPLOY FAILED: trading loop not alive on the new code"
rollback
