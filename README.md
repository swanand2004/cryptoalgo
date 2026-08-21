# cryptoalgo

A systematic crypto trading platform on Delta Exchange India, covering BTCUSD /
ETHUSD / SOLUSD perpetual futures. Ingests market data over REST + WebSocket,
runs a strategy ensemble (trend, mean-reversion, funding carry, order-book
imbalance, and a locally-trained ML model) through a shared backtest/live code
path, sizes positions with volatility targeting, and enforces hard risk limits
including a daily-loss kill switch and a Deadman Switch heartbeat for live mode.

See `C:\Users\swana\.claude\plans\melodic-beaming-badger.md` for the original
architecture plan.

## Structure

- `backend/` -- Python/FastAPI. Data ingestion, feature engineering, strategies,
  ML, backtesting, risk management, paper/live execution, REST + WebSocket API.
- `frontend/` -- React/TypeScript/Vite dashboard.

## First-time setup

**1. Rotate your Delta API key.** If you ever pasted a real key/secret into a
chat window, treat it as compromised -- regenerate it at
https://www.delta.exchange/app/account/manageapikeys before using it here.

**2. Backend**

```
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Edit `backend/.env`:
- Leave `DELTA_ENV=testnet` while developing (recommended). Market data
  endpoints work with no key at all; only order placement needs credentials.
- To enable live trading later, set `DELTA_API_KEY` / `DELTA_API_SECRET` and
  `DELTA_ENV=production`, then flip the mode toggle to Live from the
  dashboard. The app always boots in **paper mode** regardless of this file.

Run it:

```
.venv\Scripts\python -m uvicorn app.main:app --reload
```

First boot backfills ~90 days of candle history per symbol/resolution, which
takes a couple of minutes (visible in the logs). The public WebSocket starts
streaming live ticks immediately.

**3. Frontend**

```
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (use `localhost`, not `127.0.0.1` -- Vite binds
IPv6 on Windows by default). The dev server proxies `/api` and `/ws` to the
backend on port 8000.

## Running tests

```
cd backend
.venv\Scripts\python -m pytest
```

Covers HMAC signing correctness, the paper broker's fill/fee simulation, risk
limit enforcement, the backtest engine, and the ML walk-forward pipeline.

## Safety notes

- The platform **always starts in paper mode**. Live trading requires an
  explicit toggle from the dashboard every session, and is blocked entirely if
  `DELTA_API_KEY`/`DELTA_API_SECRET` aren't configured.
- The kill switch (dashboard header) immediately flattens all positions,
  forces paper mode, and halts trading. It also trips automatically if the
  configured daily loss limit (`MAX_DAILY_LOSS_USD` in `.env`) is breached.
- Before ever pointing this at production with real funds, run it against
  Delta's **testnet** (`DELTA_ENV=testnet`, get a Demo account key from
  https://demo.delta.exchange) in live mode first to confirm order placement
  behaves as expected end-to-end.
