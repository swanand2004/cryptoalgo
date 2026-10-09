#!/usr/bin/env bash
# Owner-run: deploy the latest tested code, then switch on Telegram trade alerts
# on the server. Secrets come from ~/.cryptoalgo_secrets/telegram.json (never
# from the repo). Re-running replaces the old TELEGRAM_ lines, never duplicates.
set -e
KEY=~/.ssh/cryptoalgo_vps
HOST=root@148.113.8.82
PY=~/Desktop/cryptoalgo/backend/.venv/Scripts/python.exe

echo "== Deploying the latest tested code"
bash ~/Desktop/cryptoalgo/deploy/deploy_vps.sh

echo "== Reading the bot token and chat id (local secrets file)"
TOKEN=$($PY -c "import json,os;print(json.load(open(os.path.expanduser('~/.cryptoalgo_secrets/telegram.json')))['trades_bot_token'])")
CHAT=$($PY -c "import json,os;print(json.load(open(os.path.expanduser('~/.cryptoalgo_secrets/telegram.json')))['trades_chat_id'])")
[ -n "$TOKEN" ] && [ -n "$CHAT" ] || { echo "Missing token or chat id"; exit 1; }

echo "== Switching on Telegram alerts on the server"
ssh -i $KEY -p 20024 $HOST "set -e; cd /opt/cryptoalgo/backend
cp .env .env.bak_\$(date +%Y%m%d_%H%M)
sed -i '/^TELEGRAM_BOT_TOKEN=/d; /^TELEGRAM_CHAT_ID=/d' .env
printf 'TELEGRAM_BOT_TOKEN=%s\nTELEGRAM_CHAT_ID=%s\n' '$TOKEN' '$CHAT' >> .env
chown cryptoalgo:cryptoalgo .env; chmod 600 .env
systemctl restart cryptoalgo; sleep 20
echo; echo \"SERVER STATUS: \$(systemctl is-active cryptoalgo)\""
echo
echo "Done. Test: toggle the Kill Switch ON then OFF on the website -- a Telegram message should arrive."
echo "Then go back to Claude and type: telegram done"
