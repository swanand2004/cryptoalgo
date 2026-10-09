#!/usr/bin/env bash
# Security update of the server's Python packages (run by the owner).
# Uploads backend/requirements.txt, saves the CURRENT package list first (for
# rollback), installs, restarts, shows status. Rollback: pip install -r the
# saved backups/pip_freeze_*.txt and restart.
set -e
KEY=~/.ssh/cryptoalgo_vps
HOST=root@148.113.8.82
ROOT=~/Desktop/cryptoalgo

echo "== 1/2 Uploading requirements.txt"
scp -i $KEY -P 20024 "$ROOT/backend/requirements.txt" $HOST:/tmp/requirements.txt

echo "== 2/2 Saving current packages, installing updates, restarting"
ssh -i $KEY -p 20024 $HOST 'set -e; cd /opt/cryptoalgo
TS=$(date +%Y%m%d_%H%M)
sudo -u cryptoalgo backend/.venv/bin/pip freeze > backups/pip_freeze_$TS.txt
cp backend/requirements.txt backups/requirements_$TS.txt 2>/dev/null || true
install -o cryptoalgo -g cryptoalgo -m 644 /tmp/requirements.txt backend/requirements.txt
sudo -u cryptoalgo backend/.venv/bin/pip install -q -r backend/requirements.txt
systemctl restart cryptoalgo
sleep 20
echo
echo "SERVER STATUS: $(systemctl is-active cryptoalgo)"
echo "Rollback if needed: pip install -r backups/pip_freeze_$TS.txt"'
echo
echo "If it says SERVER STATUS: active -> DONE. Go back to Claude and type: updated"
