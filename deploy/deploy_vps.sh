#!/usr/bin/env bash
# Deploy the local cryptoalgo build to the VPS (run by the owner).
# Packs backend/app (no secrets, no model stores) + frontend/dist, uploads,
# backs up the running code on the server, installs, restarts, shows status.
set -e
KEY=~/.ssh/cryptoalgo_vps
HOST=root@148.113.8.82
ROOT=~/Desktop/cryptoalgo
# Optional 1st argument: a reviewed staging copy of backend/ to ship instead of
# the working tree (used when part of the tree is still being built).
BACKEND_DIR="${1:-$ROOT/backend}"

echo "== 1/4 Packing the backend from $BACKEND_DIR"
cd "$BACKEND_DIR"
tar czf /tmp/deploy_app.tgz --exclude=__pycache__ --exclude=app/gold/model_store --exclude=app/analyst/model_store app

echo "== 2/4 Building the website"
cd "$ROOT/frontend"
npm run build
tar czf /tmp/deploy_dist.tgz dist

echo "== 3/4 Uploading to the server"
scp -i $KEY -P 20024 /tmp/deploy_app.tgz /tmp/deploy_dist.tgz $HOST:/tmp/

echo "== 4/4 Installing and restarting (old code is backed up first)"
ssh -i $KEY -p 20024 $HOST 'set -e; cd /opt/cryptoalgo
tar czf backups/code_$(date +%Y%m%d_%H%M).tgz backend/app frontend/dist
tar xzf /tmp/deploy_app.tgz -C backend --no-same-owner
rm -rf frontend/dist && tar xzf /tmp/deploy_dist.tgz -C frontend --no-same-owner
chown -R cryptoalgo:cryptoalgo backend/app frontend/dist
systemctl restart cryptoalgo
sleep 20
echo
echo "SERVER STATUS: $(systemctl is-active cryptoalgo)"'
echo
echo "If it says SERVER STATUS: active -> DONE. Go back to Claude and type: deployed"
