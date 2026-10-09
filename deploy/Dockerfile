# Trading backend + dashboard, one container image.
#
# Stage 1 builds the React dashboard; only its static dist/ is carried into
# the runtime image, so Node never ships to production.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

#
# python:3.12-slim rather than alpine on purpose: pandas/numpy/scipy/lightgbm
# all ship manylinux wheels that install in seconds here, whereas alpine's musl
# libc forces a source build of every one of them (20+ minutes, and lightgbm
# needs a compiler toolchain that doubles the image size anyway).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# libgomp is LightGBM's OpenMP runtime -- the import fails without it, and the
# failure is at import time, so the whole app refuses to start.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first so a code change does not re-install the dependency tree.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /frontend/dist ./frontend/dist
ENV FRONTEND_DIST=/app/frontend/dist

# The database lives on a mounted volume, NOT in the image. Baking it in means
# every redeploy silently resets your positions and equity history to whatever
# was captured at build time.
VOLUME ["/data"]
ENV DATABASE_URL="sqlite+aiosqlite:////data/live.db"

WORKDIR /app/backend

# The loop is the point of the container, so the health check asks whether it
# is still TRADING, not merely whether HTTP answers. A process serving 200s
# with a dead scheduler is the exact failure this system had on 2026-09-07,
# and an HTTP-only check would have called it healthy the whole time.
HEALTHCHECK --interval=120s --timeout=15s --start-period=180s --retries=3 \
  CMD python -c "import sqlite3,datetime as dt,os,sys; \
p=os.environ['DATABASE_URL'].split('///')[-1]; \
c=sqlite3.connect(p); \
t=c.execute(\"select max(timestamp) from equity_snapshots where mode='paper'\").fetchone()[0]; \
age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc)).total_seconds(); \
sys.exit(0 if age < 600 else 1)" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
