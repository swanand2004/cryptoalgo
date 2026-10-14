# Running the book 24/7 on free hosting

Measured footprint: **11 MB RAM**, **73 MB database** (after `slim_db.py`), a few
API calls a minute. This fits comfortably inside several genuinely free tiers.

---

## Read this before you deploy anything

**Free hosting is fine for paper. It is not fine for real money.** Free tiers have
no SLA, get reclaimed for idleness, and die without warning. If the host dies while
the book holds four leveraged positions, nothing is managing them — no kill switch,
no drawdown supervisor, no exit. That is survivable in paper and is not survivable
live. If you go live, pay for the VM.

**The strategy does not actually need a 24/7 process.** It reads 1h bars and holds
~9 days. What it needs is to run once per closed hourly bar. The 60-second loop is
convenience, not requirement — which is why option C works at all.

---

## Option A — Oracle Cloud Always Free (recommended)

Genuinely free forever, and far larger than anything else on this list: up to
4 ARM cores and 24 GB RAM. Card required for identity, never charged.

1. Create an **Ampere A1 (ARM)** instance, Ubuntu 22.04, 1 core / 6 GB is plenty.
2. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER && newgrp docker
   ```
3. Copy the project up, and the slimmed database into the volume:
   ```bash
   python deploy/slim_db.py backend/cryptoalgo.db deploy/live.db   # run locally
   scp -r . ubuntu@<ip>:~/cryptoalgo
   ```
4. On the server:
   ```bash
   cd ~/cryptoalgo/deploy
   cp ~/cryptoalgo/backend/.env .env       # your keys; keep it off git
   docker compose up -d --build
   docker volume ls                        # find deploy_trading-data
   docker cp live.db cryptoalgo:/data/live.db
   docker compose restart
   ```
5. Watch it trade:
   ```bash
   docker compose logs -f | grep -E "Book:|consolidated:"
   ```

**Oracle reclaims idle Always Free instances.** This app's CPU is low enough to look
idle. Keep a small periodic load or accept the reclaim risk.

## Public 24/7 site (free) — on top of Option A

The image now builds the dashboard into itself, so one container serves the
whole site. A **Cloudflare Tunnel** gives it a public HTTPS URL for free: the
VM dials *out* to Cloudflare, so no port is opened and there is no firewall rule
to get wrong.

1. In `deploy/.env` (never committed), add a password. `docker compose up`
   **refuses to start without it** — the API can switch the book to live and
   fire the kill switch, so an unprotected public URL is not an option:
   ```
   DASHBOARD_PASSWORD=<long random string>
   DASHBOARD_USER=admin            # optional, defaults to admin
   ```
2. Pick a URL:
   - **Your own fixed hostname** (needs a domain on Cloudflare's free plan):
     Cloudflare dashboard → Zero Trust → Networks → Tunnels → *Create tunnel* →
     copy the token into `deploy/.env` as `CLOUDFLARE_TUNNEL_TOKEN=...`, and add a
     public hostname pointing at `http://localhost:8000`. Then:
     ```bash
     docker compose --profile public up -d --build
     ```
   - **No domain at all** (random `*.trycloudflare.com` URL that changes on every
     restart — fine to try it, poor for a bookmark):
     ```bash
     docker compose up -d --build
     docker run -d --name quick-tunnel --network host --restart unless-stopped \
       cloudflare/cloudflared:latest tunnel --no-autoupdate --url http://localhost:8000
     docker logs quick-tunnel 2>&1 | grep trycloudflare.com
     ```
3. Open the URL, log in with the browser prompt. The login is remembered for
   30 days by a signed cookie; changing `DASHBOARD_PASSWORD` logs everyone out.
4. Point a free uptime monitor (UptimeRobot, Better Stack) at
   `https://<your-url>/api/health` — it is the one path left open without a
   password, exactly so a monitor can reach it. It proves the site is up, *not*
   that the book is trading; the container HEALTHCHECK covers that.

## Option B — Google Cloud e2-micro

Also always-free (US regions only), 1 GB RAM. Same Docker steps. Tighter on memory
but 11 MB of app leaves plenty of headroom.

## Option C — GitHub Actions, hourly (no server at all)

Free, nothing to maintain, and it matches what the strategy actually needs. The
catch is that Actions runners are **ephemeral**, so the database cannot live on
them — point `DATABASE_URL` at a free hosted Postgres (Neon, Supabase) instead.

```yaml
# .github/workflows/trade.yml
name: trading tick
on:
  schedule:
    - cron: "5 * * * *"     # 5 min after each hour, so the 1h bar has closed
  workflow_dispatch:
jobs:
  tick:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r backend/requirements.txt
      - working-directory: backend
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          DELTA_API_KEY: ${{ secrets.DELTA_API_KEY }}
          DELTA_API_SECRET: ${{ secrets.DELTA_API_SECRET }}
        run: |
          python -c "
          import asyncio
          from app.runtime import get_or_create_runtime
          from app.trading_loop import run_trading_tick
          from app.scheduler import _incremental_ladder_history
          async def main():
              rt = await get_or_create_runtime()
              await _incremental_ladder_history(rt)   # refresh signal data FIRST
              await run_trading_tick(rt)
              await rt.stop()
          asyncio.run(main())"
```

Trade-off: the daily-loss and drawdown supervisors only run hourly instead of every
15 seconds. With ATR brackets off and risk carried by position size, that is
acceptable — but it is a real reduction in how fast a bad day gets stopped.

## What NOT to use

**Render / Railway / Fly free web services sleep on inactivity.** A sleeping trading
loop is worse than no trading loop: it holds positions it has stopped managing.

---

## Automatic deploys from GitHub (the VPS)

`.github/workflows/deploy.yml` replaces running `deploy_vps.sh` by hand:

- **Claude pushes to a `claude/...` branch** (or anyone pushes to `main`): backend
  tests + frontend build, then the deploy job **waits for your one Approve**
  (the `production` environment), then uploads, installs and restarts via
  `deploy/remote_install.sh`. Merge the branch's PR afterwards to record on
  `main` what is live; the workflow itself never writes to `main`.
- **Pull requests**: tests + build only. Nothing touches the server.
- **The deploy only passes if the trading loop is alive on the new code**: a
  fresh paper equity snapshot written after the restart, checked after 150 s so
  the 60 s loop has ticked, not merely `systemctl is-active`. Otherwise it
  restores the previous code (and Python packages, if they changed), restarts,
  and the run goes red. Same if any install step fails.
- **Refused before anything changes**: a commit that does not contain the
  current `main` (it would roll back live code), and any deploy while the
  nightly retrain is running.

One-time setup, from your laptop:

```bash
# 1. A deploy-only key (never reuse ~/.ssh/cryptoalgo_vps)
ssh-keygen -t ed25519 -f ~/.ssh/cryptoalgo_deploy -N "" -C github-deploy
ssh-copy-id -i ~/.ssh/cryptoalgo_deploy.pub -p 20024 root@<server-ip>
# 2. The server's host key, so the runner cannot be pointed at an impostor
ssh-keyscan -p 20024 <server-ip>
```

3. GitHub → repo → Settings → Secrets and variables → Actions → add
   `VPS_HOST`, `VPS_PORT`, `VPS_USER` (`root`), `VPS_SSH_KEY` (contents of
   `~/.ssh/cryptoalgo_deploy`), `VPS_KNOWN_HOSTS` (output of step 2).
4. GitHub → Settings → Environments → New environment `production` →
   *Required reviewers* → add yourself.
5. **Only after step 4**: Settings → Secrets and variables → Actions →
   *Variables* → `DEPLOY_ENABLED` = `true`. This is the on-switch; until it is
   set, pushes run tests and the build but never deploy. Order matters: if the
   environment has no reviewer, GitHub deploys with no approval at all.

Redeploy `main` by hand: Actions → deploy → *Run workflow*. Revoke access any
time by deleting the key's line from `/root/.ssh/authorized_keys` on the server.

## Whichever you pick

- **Secrets**: `.env` never goes in git or the image. Use the host's secret store.
- **Never expose port 8000 directly.** Go public only through the tunnel + password above, or reach it over an SSH tunnel:
  `ssh -L 8000:127.0.0.1:8000 user@host`, then open `http://localhost:8000`.
- **Frontend**: built into the image and served by the backend; no separate host needed. (Old route: put `dist/` on Cloudflare
  Pages/Netlify — no longer necessary.)
- **Verify it is actually trading**, not merely up. The tell is the equity snapshot age:
  ```bash
  docker compose exec backend python -c "
  import sqlite3,datetime as dt
  c=sqlite3.connect('/data/live.db')
  t=c.execute(\"select max(timestamp) from equity_snapshots where mode='paper'\").fetchone()[0]
  print(t, (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)).total_seconds(), 'sec old')"
  ```
  Under ~120s means the loop is alive. Minutes or hours means it has stalled while
  still answering HTTP — the exact failure mode seen on 2026-09-07.
- **Back up the database.** It holds your positions, equity history and track record:
  `docker compose exec backend sqlite3 /data/live.db ".backup /data/backup.db"`
