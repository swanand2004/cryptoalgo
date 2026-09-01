import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type GoldPrediction } from '../lib/api'
import { signalTier } from '../lib/signalTier'
import { CandleChart } from '../components/CandleChart'

const REGIME_LABEL: Record<string, { label: string; className: string }> = {
  trending: { label: 'Trending', className: 'bg-emerald-950 text-emerald-400 border border-emerald-900' },
  mean_reverting: { label: 'Mean-Reverting', className: 'bg-sky-950 text-sky-400 border border-sky-900' },
  random_walk: { label: 'Random Walk', className: 'bg-neutral-800 text-neutral-400' },
  unknown: { label: 'Unknown', className: 'bg-neutral-800 text-neutral-500' },
}

const HORIZON_LABEL: Record<string, string> = {
  intraday: 'Intraday',
  short_term: 'Short-Term',
  long_term: 'Long-Term',
}
const HORIZON_ORDER = ['intraday', 'short_term', 'long_term']
const HORIZON_HINT: Record<string, string> = {
  intraday: '~1 hour ahead, 15m bars',
  short_term: '~1 day ahead, 1h bars',
  long_term: '~1 week ahead, 1d bars',
}

const ACTION_STYLE: Record<string, string> = {
  BUY: 'bg-emerald-600 text-emerald-50',
  SELL: 'bg-red-600 text-red-50',
  HOLD: 'bg-neutral-800 text-neutral-400',
}

// Risk is the model's own honest walk-forward track record for this
// horizon, not a generic disclaimer -- see backend/app/gold/engine.py's
// _risk_level. "very_high" covers both "below coin-flip" and "too little
// data to trust," which is deliberately where gold's short_term/long_term
// horizons currently sit.
const RISK_STYLE: Record<string, { label: string; className: string }> = {
  low: { label: 'Low risk', className: 'bg-emerald-950 text-emerald-400 border border-emerald-900' },
  medium: { label: 'Medium risk', className: 'bg-amber-950 text-amber-400 border border-amber-900' },
  high: { label: 'High risk', className: 'bg-orange-950 text-orange-400 border border-orange-900' },
  very_high: { label: 'Very high risk', className: 'bg-red-950 text-red-400 border border-red-900' },
}

const DEEP_STAGES: {
  key: string
  title: string
  hint: string
  kind: 'signed' | 'confidence'
}[] = [
  {
    key: 'deep_trend_consensus',
    title: '1. Trend Consensus',
    hint: '10 trend indicators (EMA spreads, MACD, ROC, regression slope) -- how many agree on direction',
    kind: 'signed',
  },
  {
    key: 'deep_meanrev_signal',
    title: '2. Mean-Reversion Pressure',
    hint: 'Bollinger & moving-average z-scores -- how far price sits from "fair value" and which way it should snap back',
    kind: 'signed',
  },
  {
    key: 'deep_regime_confidence',
    title: '3. Regime Confidence',
    hint: 'Efficiency ratio + ADX -- how much conviction to have at all right now (low = choppy, high = a genuine trend)',
    kind: 'confidence',
  },
  {
    key: 'deep_astro_consensus',
    title: '4. Astrology Consensus',
    hint: '6 signed planetary-transit readings -- benefic vs malefic tilt, agreement across independent systems',
    kind: 'signed',
  },
]

function SignedBar({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, value))
  const pct = Math.abs(clamped) * 50
  const positive = clamped >= 0
  return (
    <div className="relative h-2 w-full rounded-full bg-neutral-800 overflow-hidden">
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-neutral-700" />
      <div
        className={`absolute top-0 bottom-0 rounded-full transition-all duration-500 ${positive ? 'bg-emerald-500' : 'bg-red-500'}`}
        style={positive ? { left: '50%', width: `${pct}%` } : { right: '50%', width: `${pct}%` }}
      />
    </div>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="relative h-2 w-full rounded-full bg-neutral-800 overflow-hidden">
      <div className="absolute inset-y-0 left-0 rounded-full bg-amber-500 transition-all duration-500" style={{ width: `${pct}%` }} />
    </div>
  )
}

function DeepAnalysisEngine({ indicators }: { indicators: Record<string, number> }) {
  const composite = indicators['deep_composite_score']
  const tier = composite != null ? signalTier(composite) : null

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-sm font-medium text-neutral-300 mb-1">Deep Analysis Engine</div>
      <div className="text-xs text-neutral-600 mb-4">
        A 4-stage cascade re-evaluated every ~3-minute tick: each stage's own reading feeds into the next, ending in one
        regime-adaptive composite score below. Second-order analysis on top of the {Object.keys(indicators).length}
        {' '}underlying indicators, not a replacement for them.
      </div>

      <div className="space-y-3">
        {DEEP_STAGES.map((stage, i) => {
          const value = indicators[stage.key]
          return (
            <div key={stage.key} className="flex items-center gap-3">
              <div className="w-44 shrink-0">
                <div className="text-xs font-medium text-neutral-300">{stage.title}</div>
              </div>
              <div className="flex-1 group relative">
                {stage.kind === 'signed' ? <SignedBar value={value ?? 0} /> : <ConfidenceBar value={value ?? 0} />}
                <div className="absolute left-0 top-full mt-1 hidden group-hover:block text-[10px] text-neutral-500 bg-neutral-950 border border-neutral-800 rounded px-2 py-1 z-10 max-w-xs">
                  {stage.hint}
                </div>
              </div>
              <div className="w-14 shrink-0 text-right font-mono text-xs text-neutral-400 tabular-nums">
                {value != null ? value.toFixed(2) : '--'}
              </div>
              {i < DEEP_STAGES.length - 1 && <div className="w-3 shrink-0 text-center text-neutral-700">&darr;</div>}
            </div>
          )
        })}
      </div>

      <div className="mt-5 pt-4 border-t border-neutral-800 flex items-center justify-between">
        <div>
          <div className="text-xs font-medium text-neutral-400">Composite Score</div>
          <div className="text-[11px] text-neutral-600 mt-0.5">
            regime confidence &times; (trend/reversion blend) + astrology tilt
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-2xl text-neutral-100 tabular-nums">
            {composite != null ? `${composite >= 0 ? '+' : ''}${composite.toFixed(3)}` : '--'}
          </span>
          {tier && <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${tier.badgeClass}`}>{tier.label}</span>}
        </div>
      </div>
    </div>
  )
}

/** Re-renders on an interval so relative-time labels (countdowns, "Ns ago")
 * stay live without waiting for the next data refetch. */
function useNowTick(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

function secondsAgoLabel(iso: string | null | undefined, now: number): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return 'never'
  const seconds = Math.max(0, Math.round((now - then) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  return `${Math.floor(seconds / 3600)}h ago`
}

function countdownLabel(iso: string | null | undefined, now: number): string {
  if (!iso) return '--'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '--'
  return formatSecondsAway(Math.round((then - now) / 1000))
}

function formatSecondsAway(seconds: number): string {
  if (seconds <= 0) return 'now'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function formatDaysAway(daysAway: number | null | undefined): string {
  if (daysAway == null) return '--'
  return formatSecondsAway(Math.round(daysAway * 86400))
}

function formatHoursAway(hoursAway: number | null | undefined): string {
  if (hoursAway == null) return '--'
  return formatSecondsAway(Math.round(hoursAway * 3600))
}

const MACRO_EVENT_HINT: Record<string, string> = {
  FOMC: 'Fed rate decision -- moves gold via rate-path/USD-strength expectations',
  CPI: 'US inflation print -- moves gold via real-yield expectations',
  NFP: 'US jobs report -- moves gold via Fed-policy expectations',
}

/** Briefly highlights when a numeric value changes between refetches --
 * the "something is continuously happening" cue for the hero price. */
function useFlashOnChange(value: number | null | undefined): boolean {
  const previous = useRef(value)
  const [flashing, setFlashing] = useState(false)
  useEffect(() => {
    if (value != null && previous.current != null && value !== previous.current) {
      setFlashing(true)
      const id = setTimeout(() => setFlashing(false), 700)
      previous.current = value
      return () => clearTimeout(id)
    }
    previous.current = value
  }, [value])
  return flashing
}

function EngineScanBar() {
  return (
    <div className="relative h-0.5 w-full overflow-hidden rounded-full bg-neutral-800">
      <div className="gold-scan absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r from-transparent via-amber-400 to-transparent" />
    </div>
  )
}

/** How many of BTC/ETH/SOL currently support this call, given each one's
 * OWN most recent move and its correlation sign with gold (both already-
 * observable, no lookahead). Researched 2026-08-22: this contemporaneous
 * relationship has ZERO exploitable lead time for FORECASTING (checked
 * explicitly at lags from 15 minutes to 4 hours, on both 15m and 1h data --
 * none survived), so it's never used to move the model's own score. This is
 * a real-time CORROBORATION read for a human call, not a second vote baked
 * into the prediction itself. */
function cryptoConfirmation(
  action: 'BUY' | 'SELL' | 'HOLD',
  crossAsset: import('../lib/api').GoldCrossAsset | undefined
): { agree: number; disagree: number; total: number } | null {
  if (!crossAsset?.ready || action === 'HOLD') return null
  const wantSign = action === 'BUY' ? 1 : -1
  let agree = 0
  let disagree = 0
  let total = 0
  for (const asset of Object.values(crossAsset.assets)) {
    if (asset.correlation == null || asset.recent_return_pct == null) continue
    if (Math.abs(asset.correlation) < 0.15) continue // too decoupled right now to read anything from
    const impliedDirection = Math.sign(asset.correlation) * Math.sign(asset.recent_return_pct)
    total += 1
    if (impliedDirection === wantSign) agree += 1
    else if (impliedDirection === -wantSign) disagree += 1
  }
  return total > 0 ? { agree, disagree, total } : null
}

function PredictionCard({
  horizon,
  prediction,
  now,
  crossAsset,
}: {
  horizon: string
  prediction: GoldPrediction | undefined
  now: number
  crossAsset: import('../lib/api').GoldCrossAsset | undefined
}) {
  const hasScore = prediction && prediction.score !== null
  const tier = hasScore ? signalTier(prediction!.score as number) : null
  const accuracy = prediction?.mean_directional_accuracy ?? null
  const action = prediction?.action ?? 'HOLD'
  const confirmation = cryptoConfirmation(action, crossAsset)
  const risk = RISK_STYLE[prediction?.risk_level ?? 'very_high']

  return (
    <div className="rounded-md border border-neutral-800 p-4 flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-medium text-neutral-200">{HORIZON_LABEL[horizon] ?? horizon}</div>
          <div className="text-[11px] text-neutral-600">{HORIZON_HINT[horizon]}</div>
        </div>
        <span className={`shrink-0 px-2.5 py-1 rounded-md text-xs font-bold tracking-wide ${ACTION_STYLE[action]}`}>
          {action}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${risk.className}`}>{risk.label}</span>
        {prediction?.risk_reason && <span className="text-[11px] text-neutral-600">{prediction.risk_reason}</span>}
      </div>

      {!hasScore && <div className="text-neutral-700 font-mono text-sm">no model trained yet</div>}

      {hasScore && tier && (
        <>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${tier.badgeClass}`}>{tier.label}</span>
            <span className="font-mono text-xs text-neutral-600">
              {(prediction!.score as number) >= 0 ? '+' : ''}
              {(prediction!.score as number).toFixed(3)}
            </span>
          </div>

          {prediction!.target_price != null && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <div className="text-neutral-600">Target price</div>
                <div className="font-mono text-neutral-100 text-sm">
                  ${prediction!.target_price!.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </div>
              </div>
              <div>
                <div className="text-neutral-600">Expected move</div>
                <div className={`font-mono text-sm ${(prediction!.expected_move_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(prediction!.expected_move_pct ?? 0) >= 0 ? '+' : ''}
                  {(prediction!.expected_move_pct ?? 0).toFixed(2)}%
                </div>
              </div>
            </div>
          )}

          <div className="text-[11px] text-neutral-600">
            valid until <span className="text-neutral-400">{countdownLabel(prediction!.valid_until, now)}</span>
          </div>

          {confirmation && (
            <div
              className={`text-[11px] ${
                confirmation.agree > confirmation.disagree
                  ? 'text-emerald-400'
                  : confirmation.disagree > confirmation.agree
                  ? 'text-red-400'
                  : 'text-neutral-500'
              }`}
            >
              crypto right now: {confirmation.agree}/{confirmation.total} confirming this call
              {confirmation.disagree > 0 && `, ${confirmation.disagree} against`}
            </div>
          )}

          {!prediction!.decidable && (
            <div className="text-[11px] text-neutral-600 italic">abstaining -- not confident enough to call</div>
          )}
          {accuracy != null && (
            <div className="text-[11px] text-neutral-600">
              walk-forward accuracy: {(accuracy * 100).toFixed(1)}%
              {prediction?.n_samples != null && ` (n=${prediction.n_samples})`}
            </div>
          )}
        </>
      )}
    </div>
  )
}

const SEASONALITY_STYLE: Record<string, string> = {
  positive: 'text-emerald-400',
  negative: 'text-red-400',
  neutral: 'text-neutral-400',
}

function EventsPanel({ events }: { events: import('../lib/api').GoldEvents | undefined }) {
  if (!events) return null
  const { seasonality, gold_buying_tithis, next_macro_event } = events

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-sm font-medium text-neutral-300 mb-3">Calendar & Events</div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {next_macro_event.event && (
          <div className="rounded-md border border-amber-900/50 bg-amber-950/20 p-3">
            <div className="text-xs font-medium text-neutral-400 mb-1">Next US Macro Event</div>
            <div className="text-sm font-semibold text-amber-400">
              {next_macro_event.event} in {formatHoursAway(next_macro_event.hours_away)}
            </div>
            <div className="text-[11px] text-neutral-600 mt-1">{MACRO_EVENT_HINT[next_macro_event.event]}</div>
          </div>
        )}
        <div className="rounded-md border border-neutral-800 p-3">
          <div className="text-xs font-medium text-neutral-400 mb-1">This Month's Documented Seasonality</div>
          <div className={`text-sm font-semibold ${SEASONALITY_STYLE[seasonality.effect]}`}>
            {seasonality.month}: {seasonality.effect}
          </div>
          <div className="text-[11px] text-neutral-600 mt-1">{seasonality.note}</div>
          <div className="text-[10px] text-neutral-700 mt-2 italic">{seasonality.source}</div>
        </div>
        {(['shukla_tritiya', 'krishna_trayodashi'] as const).map((key) => {
          const t = gold_buying_tithis[key]
          return (
            <div key={key} className="rounded-md border border-neutral-800 p-3">
              <div className="text-xs font-medium text-neutral-400 mb-1">{t.label}</div>
              <div className="text-sm font-semibold text-amber-400">{formatDaysAway(t.days_away)}</div>
              <div className="text-[11px] text-neutral-600 mt-1">{t.date ?? '--'}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const CORRELATION_LABEL: Record<string, { label: string; className: string }> = {
  moving_with: { label: 'Moving WITH gold', className: 'text-emerald-400' },
  loosely_with: { label: 'Loosely with gold', className: 'text-emerald-500/70' },
  decoupled: { label: 'Decoupled', className: 'text-neutral-500' },
  loosely_against: { label: 'Loosely against gold', className: 'text-red-500/70' },
  moving_against: { label: 'Moving AGAINST gold', className: 'text-red-400' },
  unknown: { label: 'Unknown', className: 'text-neutral-600' },
}

function CrossAssetPanel({ crossAsset }: { crossAsset: import('../lib/api').GoldCrossAsset | undefined }) {
  if (!crossAsset?.ready) return null

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-sm font-medium text-neutral-300 mb-1">Crypto Correlation</div>
      <div className="text-xs text-neutral-600 mb-4">
        Rolling {crossAsset.window_bars}-bar ({crossAsset.resolution}) correlation with BTC/ETH/SOL -- a real, strongly
        significant SAME-INSTANT relationship (researched 2026-08-22: both react to shared risk-sentiment/liquidity
        conditions), but with ZERO exploitable lead time in either direction. Context for reading the current move, not
        a forecasting input -- never fed to the prediction models above.
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {Object.entries(crossAsset.assets).map(([symbol, asset]) => {
          const style = CORRELATION_LABEL[asset.interpretation] ?? CORRELATION_LABEL.unknown
          return (
            <div key={symbol} className="rounded-md border border-neutral-800 p-3">
              <div className="text-xs font-medium text-neutral-400 mb-1">{symbol}</div>
              <div className={`text-sm font-semibold ${style.className}`}>{style.label}</div>
              <div className="text-[11px] text-neutral-600 mt-1 font-mono">
                r = {asset.correlation != null ? asset.correlation.toFixed(2) : '--'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function Gold() {
  const now = useNowTick(1000)
  const snapshotQuery = useQuery({
    queryKey: ['gold_snapshot'],
    queryFn: api.goldSnapshot,
    refetchInterval: 30_000,
  })
  const candlesQuery = useQuery({
    queryKey: ['gold_candles'],
    queryFn: () => api.goldCandles('15m', 300),
    refetchInterval: 60_000,
  })

  const snapshot = snapshotQuery.data
  const regime = REGIME_LABEL[snapshot?.regime ?? 'unknown']
  const priceFlash = useFlashOnChange(snapshot?.price ?? null)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <style>{`
        @keyframes gold-scan-move { 0% { left: -33%; } 100% { left: 100%; } }
        .gold-scan { animation: gold-scan-move 2.6s linear infinite; }
        @keyframes gold-flash { 0% { color: #fbbf24; } 100% { color: inherit; } }
        .gold-flash { animation: gold-flash 700ms ease-out; }
      `}</style>

      <div className="rounded-lg border border-neutral-800 bg-gradient-to-b from-neutral-900 to-neutral-900/60 p-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-500" />
              </span>
              <div className="text-sm font-medium text-neutral-200">Gold Research Engine</div>
              <span className="text-[10px] text-neutral-600 uppercase tracking-wider">PAXG / Paxos Gold</span>
            </div>
            <div className="text-xs text-neutral-600 mt-1 max-w-xl">
              Standalone research page -- NOT paper-traded, no positions or PnL. A dedicated feature bank (technicals,
              Fibonacci/regression channels, financial astrology) and 3 independently-trained horizon models, entirely
              separate from the crypto strategies.
            </div>
          </div>
          <div className="text-right">
            <div className={`font-mono text-3xl text-neutral-100 tabular-nums ${priceFlash ? 'gold-flash' : ''}`}>
              {snapshot?.price != null ? `$${snapshot.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '--'}
            </div>
            <div className="text-xs text-neutral-600">{secondsAgoLabel(snapshot?.as_of, now)}</div>
          </div>
        </div>

        <div className="mt-3">
          <EngineScanBar />
        </div>

        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs text-neutral-500">Regime:</span>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${regime.className}`}>{regime.label}</span>
        </div>
      </div>

      {!snapshot?.ready && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-6 text-center text-neutral-600 text-sm">
          Waiting for gold data to backfill and the first research tick to run (every ~3 minutes)...
        </div>
      )}

      {snapshot?.ready && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {HORIZON_ORDER.map((horizon) => (
              <PredictionCard
                key={horizon}
                horizon={horizon}
                prediction={snapshot.predictions[horizon]}
                now={now}
                crossAsset={snapshot.cross_asset}
              />
            ))}
          </div>

          <EventsPanel events={snapshot.events} />
          <CrossAssetPanel crossAsset={snapshot.cross_asset} />

          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
            <div className="text-sm font-medium text-neutral-300 mb-3">Price (15m)</div>
            <CandleChart candles={candlesQuery.data ?? []} />
          </div>

          <DeepAnalysisEngine indicators={snapshot.indicators} />
        </>
      )}
    </div>
  )
}
