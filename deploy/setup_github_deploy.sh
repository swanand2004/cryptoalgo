#!/usr/bin/env bash
# One-time setup for .github/workflows/deploy.yml (run by the owner, on the
# laptop, in Git Bash). Nothing secret leaves this machine except into
# GitHub's encrypted secret store.
#
#   bash deploy/setup_github_deploy.sh <server-ip> [ssh-port] [ssh-user]
#
# Needs: the GitHub CLI, logged in (winget install GitHub.cli, then
# `gh auth login`), and your existing server key ~/.ssh/cryptoalgo_vps.
#
# What it does, each step safe to re-run:
#   1. creates a DEPLOY-ONLY key ~/.ssh/cryptoalgo_deploy (your personal key
#      never goes to GitHub)
#   2. authorises it on the server, using your existing key
#   3. proves the new key can log in
#   4. pins the server's host key from your own known_hosts (what you already
#      trust), falling back to a fresh scan only if there is none
#   5. stores VPS_HOST, VPS_PORT, VPS_USER, VPS_SSH_KEY, VPS_KNOWN_HOSTS as
#      repository secrets
#   6. last, sets DEPLOY_ENABLED=true -- only after everything above worked
#
# Undo: delete the "github-deploy" line from ~/.ssh/authorized_keys on the
# server, and `gh variable delete DEPLOY_ENABLED`.
set -euo pipefail

HOST=${1:?usage: bash deploy/setup_github_deploy.sh <server-ip> [ssh-port] [ssh-user]}
PORT=${2:-20024}
USER_=${3:-root}
ADMIN_KEY=~/.ssh/cryptoalgo_vps
DEPLOY_KEY=~/.ssh/cryptoalgo_deploy

say() { echo; echo "== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

for tool in gh ssh ssh-keygen ssh-keyscan; do
  command -v "$tool" >/dev/null || die "'$tool' not found. Install it and re-run."
done
gh auth status >/dev/null 2>&1 || die "GitHub CLI is not logged in. Run: gh auth login"
[[ -f $ADMIN_KEY ]] || die "$ADMIN_KEY not found (the key deploy_vps.sh uses)."

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null) \
  || die "Run this from inside your cryptoalgo checkout."
echo "Repository: $REPO   Server: $USER_@$HOST port $PORT"

say "1/6 Deploy-only key"
if [[ -f $DEPLOY_KEY ]]; then
  echo "exists: $DEPLOY_KEY"
else
  ssh-keygen -q -t ed25519 -f "$DEPLOY_KEY" -N "" -C github-deploy
  echo "created: $DEPLOY_KEY"
fi
PUB=$(cat "$DEPLOY_KEY.pub")

say "2/6 Authorising it on the server (using $ADMIN_KEY)"
printf '%s\n' "$PUB" | ssh -i "$ADMIN_KEY" -p "$PORT" "$USER_@$HOST" '
  umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys
  read -r key
  grep -qxF "$key" ~/.ssh/authorized_keys && echo "already authorised" \
    || { printf "%s\n" "$key" >> ~/.ssh/authorized_keys; echo "added"; }'

say "3/6 Testing a login with the new key"
ssh -i "$DEPLOY_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -p "$PORT" "$USER_@$HOST" \
  'echo "ok: logged in as $(whoami) on $(hostname)"' \
  || die "the new key cannot log in; nothing was stored in GitHub."

say "4/6 Pinning the server's host key"
if [[ $PORT == 22 ]]; then LOOKUP=$HOST; else LOOKUP="[$HOST]:$PORT"; fi
KNOWN=$(ssh-keygen -F "$LOOKUP" 2>/dev/null | grep -v '^#' || true)
if [[ -n $KNOWN ]]; then
  echo "using the entry already in your known_hosts (what your laptop trusts)"
else
  echo "no known_hosts entry; scanning the server now"
  KNOWN=$(ssh-keyscan -p "$PORT" "$HOST" 2>/dev/null)
fi
[[ -n $KNOWN ]] || die "could not get the server's host key."

say "5/6 Storing secrets in $REPO"
put() { printf '%s' "$2" | gh secret set "$1" -R "$REPO" >/dev/null && echo "set $1"; }
put VPS_HOST "$HOST"
put VPS_PORT "$PORT"
put VPS_USER "$USER_"
put VPS_SSH_KEY "$(cat "$DEPLOY_KEY")"
put VPS_KNOWN_HOSTS "$KNOWN"

say "6/6 Turning deploys on"
gh variable set DEPLOY_ENABLED -R "$REPO" --body true >/dev/null && echo "set DEPLOY_ENABLED=true"

echo
echo "DONE. To deploy: GitHub -> Actions -> deploy -> Run workflow -> pick the branch -> Run."
