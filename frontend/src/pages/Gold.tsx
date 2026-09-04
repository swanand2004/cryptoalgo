import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type GoldBlend, type GoldPrediction, type GoldTradePlan } from '../lib/api'
import { signalTier } from '../lib/signalTier'
import { CandleChart } from '../components/CandleChart'

// Layout note. This page used to render all three horizons as three dense
// parallel cards, each stacking a score bar, a blend panel, a trade plan and
// a levels panel -- roughly eighteen competing blocks on first paint, none of
// them louder than any other. The information was all correct and none of it
// was readable.
//
// It is now an instrument console: ONE horizon in focus at a time, with the
// other two reduced to selector chips that still show their call, and every
// secondary panel behind a closed <details>. Nothing was deleted -- the
// density moved off the default view and behind one click, which is what
// "less complicated" has to mean for a page whose whole job is showing
// numbers.

const REGIME_LABEL: Record<string, { label: string; className: string }> = {
  trending: { label: 'Trending', className: 'text-emerald-400 border-emerald-900/70 bg-emerald-950/30' },
  mean_reverting: { label: 'Mean-Reverting', className: 'text-sky-400 border-sky-900/70 bg-sky-950/30' },
  random_walk: { label: 'Random Walk', className: 'text-neutral-400 border-neutral-800 bg-neutral-900' },
  unknown: { label: 'Unknown', className: 'text-neutral-500 border-neutral-800 bg-neutral-900' },
}

const HORIZON_LABEL: Record<string, string> = {
  intraday: 'Intraday',
  short_term: 'Short-Term',
  long_term: 'Long-Term',
}
const HORIZON_ORDER = ['intraday', 'short_term', 'long_term']
const HORIZON_HINT: Record<string, string> = {
  intraday: '~1 hour ahead · 15m bars',
  short_term: '~1 day ahead · 1h bars',
  long_term: '~1 week ahead · 1d bars',
}

const ACTION_TONE: Record<string, { text: string; ring: string; glow: string; dot: string }> = {
  BUY: {
    text: 'text-emerald-400',
    ring: 'border-emerald-800/60',
    glow: 'shadow-[0_0_28px_-8px_rgba(75,185,138,0.5)]',
    dot: 'bg-emerald-400',
  },
  SELL: {
    text: 'text-red-400',
    ring: 'border-red-900/60',
    glow: 'shadow-[0_0_28px_-8px_rgba(226,100,95,0.5)]',
    dot: 'bg-red-400',
  },
  HOLD: {
    text: 'text-neutral-400',
    ring: 'border-neutral-800',
    glow: '',
    dot: 'bg-neutral-600',
  },
}

const RISK_STYLE: Record<string, { label: string; className: string }> = {
  low: { label: 'Low risk', className: 'text-emerald-400 border-emerald-900/70' },
  medium: { label: 'Medium risk', className: 'text-amber-400 border-amber-900/70' },
  high: { label: 'High risk', className: 'text-orange-400 border-orange-900/70' },
  very_high: { label: 'Very high risk', className: 'text-red-400 border-red-900/70' },
}

const DEEP_STAGES: { key: string; title: string; hint: string; kind: 'signed' | 'confidence' }[] = [
  {
    key: 'deep_trend_consensus',
    title: 'Trend consensus',
    hint: '10 trend indicators (EMA spreads, MACD, ROC, regression slope) -- how many agree on direction',
    kind: 'signed',
  },
  {
    key: 'deep_meanrev_signal',
    title: 'Mean-reversion pressure',
    hint: 'Bollinger & moving-average z-scores -- how far price sits from "fair value" and which way it should snap back',
    kind: 'signed',
  },
  {
    key: 'deep_regime_confidence',
    title: 'Regime confidence',
    hint: 'Efficiency ratio + ADX -- how much conviction to have at all right now (low = choppy, high = a genuine trend)',
    kind: 'confidence',
  },
  {
    key: 'deep_astro_consensus',
    title: 'Astrology consensus',
    hint: '6 signed planetary-transit readings -- benefic vs malefic tilt, agreement across independent systems',
    kind: 'signed',
  },
]

const MICRO_LABEL: Record<string, string> = {
  vol_term_structure: 'Realized-vol term structure',
  session: 'Trading session',
  london_fix: 'LBMA auction proximity',
  macro_event: 'US release proximity',
}

const FACTOR_LABEL: Record<string, string> = {
  real_rate_change: 'Real rates (10y TIPS)',
  dollar_change: 'Dollar (DXY)',
  breakeven_inflation: 'Breakeven inflation',
  cftc_positioning: 'CFTC positioning',
}

const MACRO_EVENT_HINT: Record<string, string> = {
  FOMC: 'Fed rate decision -- moves gold via rate-path/USD-strength expectations',
  CPI: 'US inflation print -- moves gold via real-yield expectations',
  NFP: 'US jobs report -- moves gold via Fed-policy expectations',
}

const SEASONALITY_STYLE: Record<string, string> = {
  positive: 'text-emerald-400',
  negative: 'text-red-400',
  neutral: 'text-neutral-400',
}

const CORRELATION_LABEL: Record<string, { label: string; className: string }> = {
  moving_with: { label: 'Moving WITH gold', className: 'text-emerald-400' },
  loosely_with: { label: 'Loosely with gold', className: 'text-emerald-500/70' },
  decoupled: { label: 'Decoupled', className: 'text-neutral-500' },
  loosely_against: { label: 'Loosely against gold', className: 'text-red-500/70' },
  moving_against: { label: 'Moving AGAINST gold', className: 'text-red-400' },
  unknown: { label: 'Unknown', className: 'text-neutral-600' },
}

// ---------------------------------------------------------------- helpers

const money = (v: number | null | undefined, digits = 2) =>
  v == null ? '--' : `$${v.toLocaleString(undefined, { maximumFractionDigits: digits })}`

const signedPct = (v: number | null | undefined, digits = 2) =>
  v == null ? '--' : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`

/** Re-renders on an interval so relative-time labels stay live without
 * waiting for the next data refetch. */
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

function formatSecondsAway(seconds: number): string {
  if (seconds <= 0) return 'now'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}

function countdownLabel(iso: string | null | undefined, now: number): string {
  if (!iso) return '--'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '--'
  return formatSecondsAway(Math.round((then - now) / 1000))
}

const formatDaysAway = (d: number | null | undefined) =>
  d == null ? '--' : formatSecondsAway(Math.round(d * 86400))
const formatHoursAway = (h: number | null | undefined) =>
  h == null ? '--' : formatSecondsAway(Math.round(h * 3600))

/** Briefly highlights when a numeric value changes between refetches -- the
 * "something is continuously happening" cue for the hero price. */
function useFlashOnChange(value: number | null | undefined): 'up' | 'down' | null {
  const previous = useRef(value)
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  useEffect(() => {
    if (value != null && previous.current != null && value !== previous.current) {
      setFlash(value > previous.current ? 'up' : 'down')
      const id = setTimeout(() => setFlash(null), 800)
      previous.current = value
      return () => clearTimeout(id)
    }
    previous.current = value
  }, [value])
  return flash
}

/** How many of BTC/ETH/SOL currently support this call, given each one's OWN
 * most recent move and its correlation sign with gold. Researched 2026-08-22:
 * this contemporaneous relationship has ZERO exploitable lead time for
 * FORECASTING (checked at lags from 15 minutes to 4 hours, on both 15m and 1h
 * data -- none survived), so it never moves the model's own score. A
 * real-time corroboration read for a human, not a second vote. */
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
    if (Math.abs(asset.correlation) < 0.15) continue
    const implied = Math.sign(asset.correlation) * Math.sign(asset.recent_return_pct)
    total += 1
    if (implied === wantSign) agree += 1
    else if (implied === -wantSign) disagree += 1
  }
  return total > 0 ? { agree, disagree, total } : null
}

// ------------------------------------------------------------ primitives

function Section({
  title,
  hint,
  children,
  defaultOpen = false,
}: {
  title: string
  hint?: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details open={defaultOpen} className="group rounded-lg border border-neutral-800/80 bg-neutral-950/40">
      <summary className="flex cursor-pointer list-none items-center gap-2.5 px-4 py-3 select-none">
        <span className="text-neutral-700 transition-transform group-open:rotate-90">&#9656;</span>
        <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-neutral-400">{title}</span>
        {hint && <span className="ml-auto truncate text-[10px] text-neutral-600">{hint}</span>}
      </summary>
      <div className="border-t border-neutral-800/60 px-4 py-4">{children}</div>
    </details>
  )
}

function SignedBar({ value }: { value: number }) {
  const clamped = Math.max(-1, Math.min(1, value))
  const pct = Math.abs(clamped) * 50
  const positive = clamped >= 0
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-neutral-900 ring-1 ring-inset ring-neutral-800">
      <div className="absolute bottom-0 left-1/2 top-0 w-px bg-neutral-700" />
      <div
        className={`absolute bottom-0 top-0 rounded-full transition-all duration-700 ${
          positive ? 'bg-emerald-500' : 'bg-red-500'
        }`}
        style={positive ? { left: '50%', width: `${pct}%` } : { right: '50%', width: `${pct}%` }}
      />
    </div>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-neutral-900 ring-1 ring-inset ring-neutral-800">
      <div className="absolute inset-y-0 left-0 rounded-full bg-amber-500 transition-all duration-700" style={{ width: `${pct}%` }} />
    </div>
  )
}

/** Signal score as a swept arc. A number in [-1,1] is abstract; an needle
 * sitting just off centre reads instantly as "barely an opinion", which is
 * what most of these scores honestly are. */
function SignalDial({ score, action }: { score: number | null; action: 'BUY' | 'SELL' | 'HOLD' }) {
  const s = Math.max(-1, Math.min(1, score ?? 0))
  const R = 62
  const CX = 78
  const CY = 74
  const angle = (deg: number) => ((deg - 180) * Math.PI) / 180
  const pt = (deg: number, r: number) => ({ x: CX + r * Math.cos(angle(deg)), y: CY + r * Math.sin(angle(deg)) })
  const scoreDeg = ((s + 1) / 2) * 180
  const start = pt(90, R)
  const end = pt(scoreDeg, R)
  const large = Math.abs(scoreDeg - 90) > 180 ? 1 : 0
  const sweep = scoreDeg >= 90 ? 1 : 0
  const stroke = action === 'BUY' ? '#4bb98a' : action === 'SELL' ? '#e2645f' : '#6b7280'
  const needle = pt(scoreDeg, R - 4)

  return (
    <svg viewBox="0 0 156 92" className="w-full max-w-[220px]">
      <defs>
        <filter id="dial-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {/* track */}
      <path d={`M ${pt(0, R).x} ${pt(0, R).y} A ${R} ${R} 0 0 1 ${pt(180, R).x} ${pt(180, R).y}`}
        fill="none" stroke="#1e2229" strokeWidth="9" strokeLinecap="round" />
      {/* tick marks every 22.5deg */}
      {[0, 22.5, 45, 67.5, 112.5, 135, 157.5, 180].map((d) => {
        const a = pt(d, R - 8)
        const b = pt(d, R - 12)
        return <line key={d} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#2b313b" strokeWidth="1" />
      })}
      {/* the zero line, drawn brighter than the rest: it is the only
          reference that matters on a page where scores hover near it */}
      <line x1={pt(90, R + 6).x} y1={pt(90, R + 6).y} x2={pt(90, R - 14).x} y2={pt(90, R - 14).y}
        stroke="#4b5563" strokeWidth="1.5" />
      {score != null && Math.abs(s) > 0.005 && (
        <path
          d={`M ${start.x} ${start.y} A ${R} ${R} 0 ${large} ${sweep} ${end.x} ${end.y}`}
          fill="none" stroke={stroke} strokeWidth="9" strokeLinecap="round" filter="url(#dial-glow)"
        />
      )}
      {score != null && (
        <>
          <line x1={CX} y1={CY} x2={needle.x} y2={needle.y} stroke={stroke} strokeWidth="1.5" opacity="0.65" />
          <circle cx={CX} cy={CY} r="2.5" fill={stroke} />
        </>
      )}
      <text x={CX} y={CY - 18} textAnchor="middle" className="fill-neutral-100 font-mono" style={{ fontSize: 20 }}>
        {score == null ? '--' : `${s >= 0 ? '+' : ''}${s.toFixed(2)}`}
      </text>
      <text x={6} y={88} className="fill-neutral-700" style={{ fontSize: 8, letterSpacing: '0.1em' }}>SHORT</text>
      <text x={150} y={88} textAnchor="end" className="fill-neutral-700" style={{ fontSize: 8, letterSpacing: '0.1em' }}>LONG</text>
    </svg>
  )
}

/** Stop, entry, current price and target on a real price axis, so risk and
 * reward are compared by LENGTH rather than by reading two numbers and doing
 * the subtraction. A 0.34x trade looks like a 0.34x trade.
 *
 * The current price is drawn as its own marker because entry is no longer
 * always "here": when chasing pays badly the plan quotes the limit price that
 * makes the ratio work, and the gap between NOW and THAT is the whole point. */
function PriceLadder({ trade }: { trade: GoldTradePlan }) {
  const { entry, stop_loss: stop, take_profit: tp, current_price: now } = trade
  if (entry == null || stop == null) return null

  const points = [stop, entry, ...(tp != null && !trade.model_conflict ? [tp] : []), ...(now != null ? [now] : [])]
  const lo = Math.min(...points)
  const hi = Math.max(...points)
  const span = hi - lo || 1
  const pad = span * 0.08
  const place = (v: number) => ((v - (lo - pad)) / (span + pad * 2)) * 100

  const riskLeft = Math.min(place(entry), place(stop))
  const riskWidth = Math.abs(place(entry) - place(stop))
  const rewardLeft = tp == null ? 0 : Math.min(place(entry), place(tp))
  const rewardWidth = tp == null ? 0 : Math.abs(place(tp) - place(entry))

  const marks: { pos: number; label: string; value: number; color: string; strong?: boolean }[] = [
    { pos: place(stop), label: 'stop', value: stop, color: '#e2645f' },
    { pos: place(entry), label: trade.entry_type === 'limit' ? 'limit entry' : 'entry', value: entry, color: '#f2cd7a', strong: true },
  ]
  if (tp != null && !trade.model_conflict) marks.push({ pos: place(tp), label: 'target', value: tp, color: '#4bb98a' })

  return (
    <div className="pt-1">
      <div className="relative h-2 w-full rounded-full bg-neutral-900 ring-1 ring-inset ring-neutral-800">
        <div className="absolute inset-y-0 rounded-l-full bg-red-500/40" style={{ left: `${riskLeft}%`, width: `${riskWidth}%` }} />
        {rewardWidth > 0 && !trade.model_conflict && (
          <div className="absolute inset-y-0 rounded-r-full bg-emerald-500/40" style={{ left: `${rewardLeft}%`, width: `${rewardWidth}%` }} />
        )}
        {marks.map((m) => (
          <div key={m.label} className="absolute -top-1 h-4 w-px" style={{ left: `${m.pos}%`, background: m.color }} />
        ))}
        {now != null && (
          <div
            className="absolute -top-2.5 flex -translate-x-1/2 flex-col items-center"
            style={{ left: `${place(now)}%` }}
            title={`current price ${now}`}
          >
            <span className="text-[8px] leading-none text-neutral-300">&#9660;</span>
          </div>
        )}
      </div>

      <div className="relative mt-2 h-9">
        {marks.map((m) => (
          <div key={m.label} className="absolute -translate-x-1/2 text-center" style={{ left: `${Math.max(9, Math.min(91, m.pos))}%` }}>
            <div className="text-[8px] uppercase tracking-[0.12em]" style={{ color: m.color, opacity: 0.8 }}>{m.label}</div>
            <div
              className={`font-mono tabular-nums ${m.strong ? 'text-[13px]' : 'text-[11px]'}`}
              style={{ color: m.strong ? '#f2cd7a' : m.color }}
            >
              {money(m.value)}
            </div>
          </div>
        ))}
      </div>

      {now != null && trade.entry_type === 'limit' && (
        <div className="mt-0.5 text-center text-[10px] text-neutral-500">
          now {money(now)} &middot; entry is{' '}
          <span className="font-mono tabular-nums text-amber-400">{signedPct(trade.entry_distance_pct)}</span> from here
        </div>
      )}
    </div>
  )
}

/** Measured accuracy against the bar that actually matters.
 *
 * Not 50%. Gold drifts up, so a coin flip is the wrong reference: over the
 * training sample 51-55% of moved bars closed higher, which means a model
 * scoring 52% is worse than doing nothing while looking like an edge. The
 * marker is placed against the LONG-ONLY baseline, with the +-2 sigma band
 * drawn around it, so "beats holding gold" is what the picture answers.
 *
 * The accuracy itself now excludes flat bars. PAXG prints 10.7% exactly-zero
 * 4-bar returns at 15m; counting those as errors understated this model by
 * about ten points and made a 55.8% model read as a 50.9% coin flip. */
function AccuracyStrip({
  accuracy,
  samples,
  baseline,
}: {
  accuracy: number | null
  samples: number | null
  baseline: number | null
}) {
  if (accuracy == null || !samples) return null
  const pct = accuracy * 100
  const base = baseline != null ? baseline * 100 : 50
  const bandPts = (50 / Math.sqrt(samples)) * 2
  const lo = Math.min(pct, base) - Math.max(3, bandPts * 2)
  const hi = Math.max(pct, base) + Math.max(3, bandPts * 2)
  const place = (v: number) => ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100
  const edge = pct - base
  const beats = edge > bandPts
  const color = beats ? '#4bb98a' : Math.abs(edge) <= bandPts ? '#8b8578' : '#e2645f'

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[9px] uppercase tracking-[0.16em] text-neutral-500">vs holding gold</span>
        <span className="font-mono text-[11px] tabular-nums" style={{ color }}>
          {edge >= 0 ? '+' : ''}
          {edge.toFixed(1)} <span className="text-neutral-600">&plusmn;{bandPts.toFixed(1)}</span>
        </span>
      </div>
      <div className="relative h-5 w-full overflow-hidden rounded bg-neutral-900 ring-1 ring-inset ring-neutral-800">
        <div
          className="absolute inset-y-0 bg-neutral-700/25"
          style={{ left: `${place(base - bandPts)}%`, width: `${place(base + bandPts) - place(base - bandPts)}%` }}
        />
        <div className="absolute inset-y-0 w-px bg-neutral-500" style={{ left: `${place(base)}%` }} />
        <div
          className="absolute top-1/2 h-3 w-[3px] -translate-y-1/2 rounded-full transition-all duration-700"
          style={{ left: `${place(pct)}%`, background: color, boxShadow: `0 0 10px ${color}` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[9px] tabular-nums text-neutral-600">
        <span>model {pct.toFixed(1)}%</span>
        <span>long-only {base.toFixed(1)}%</span>
      </div>
      <div className="mt-1 text-[10px] leading-snug text-neutral-600">
        {beats
          ? `Beats holding by ${edge.toFixed(1)} points on ${samples.toLocaleString()} walk-forward samples.`
          : Math.abs(edge) <= bandPts
          ? `Indistinguishable from holding gold on ${samples.toLocaleString()} samples.`
          : `WORSE than simply holding gold, by ${Math.abs(edge).toFixed(1)} points.`}
      </div>
    </div>
  )
}

/** Where the call came from, as one strip rather than a stack of panels. */
function EvidenceStrip({ blend }: { blend: GoldBlend | null | undefined }) {
  if (!blend) return null
  const rows: { label: string; value: number; note?: string }[] = [
    { label: 'Model', value: blend.model_score, note: `${Math.round(blend.model_weight * 100)}% weight` },
    { label: 'Macro', value: blend.factor_score, note: `${Math.round(blend.factor_weight * 100)}% weight` },
  ]
  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-2.5">
          <span className="w-12 shrink-0 text-[10px] uppercase tracking-wider text-neutral-500">{r.label}</span>
          <div className="flex-1">
            <SignedBar value={r.value} />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-400">
            {r.value >= 0 ? '+' : ''}
            {r.value.toFixed(2)}
          </span>
          <span className="w-16 shrink-0 text-right text-[9px] text-neutral-700">{r.note}</span>
        </div>
      ))}
      <div className="flex items-center gap-2.5 border-t border-neutral-800/70 pt-2">
        <span className="w-12 shrink-0 text-[10px] font-medium uppercase tracking-wider text-neutral-300">Blend</span>
        <div className="flex-1">
          <SignedBar value={blend.blended_score} />
        </div>
        <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-100">
          {blend.blended_score >= 0 ? '+' : ''}
          {blend.blended_score.toFixed(2)}
        </span>
        <span className="w-16 shrink-0 text-right text-[9px]">
          {blend.micro_conviction < 0.99 ? (
            <span className="text-amber-400">&times;{blend.micro_conviction.toFixed(2)}</span>
          ) : (
            <span className="text-neutral-700">full size</span>
          )}
        </span>
      </div>
      <div
        className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] ${
          blend.sides_agree ? 'bg-emerald-950/50 text-emerald-400' : 'bg-amber-950/50 text-amber-400'
        }`}
      >
        <span className={`h-1 w-1 rounded-full ${blend.sides_agree ? 'bg-emerald-400' : 'bg-amber-400'}`} />
        {blend.sides_agree ? 'model and macro agree' : 'model and macro disagree'}
      </div>
    </div>
  )
}

function FactorTable({ blend }: { blend: GoldBlend }) {
  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div>
        <div className="mb-2 text-[10px] uppercase tracking-[0.16em] text-neutral-500">Macro · votes on direction</div>
        <div className="space-y-1.5">
          {blend.factors.map((f) => (
            <div key={f.name} className="flex items-center justify-between text-[11px]" title={f.detail}>
              <span className={f.contributes ? 'text-neutral-400' : 'italic text-neutral-600'}>
                {FACTOR_LABEL[f.name] ?? f.name}
                {!f.contributes && <span className="text-neutral-700"> · context only</span>}
              </span>
              <span
                className={`font-mono tabular-nums ${
                  !f.contributes
                    ? 'text-neutral-600'
                    : f.direction === 'bullish'
                    ? 'text-emerald-400'
                    : f.direction === 'bearish'
                    ? 'text-red-400'
                    : 'text-neutral-500'
                }`}
              >
                {f.direction}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2.5 text-[10px] leading-relaxed text-neutral-600">
          Real rates 54.4% (21d) and dollar 52.9% (10d) against a 50% coin flip, &plusmn;2.1pt band. The other
          two scored at or below 50%, so they are shown and not weighted. An equal-weight mix of all four
          scored 44.6% &mdash; worse than its own best component.
        </p>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Micro · scales size only</span>
          <span
            className={`font-mono text-[11px] tabular-nums ${
              blend.micro_conviction < 0.99 ? 'text-amber-400' : 'text-neutral-600'
            }`}
          >
            &times;{blend.micro_conviction.toFixed(2)}
          </span>
        </div>
        <div className="space-y-1.5">
          {blend.micro_factors?.map((m) => (
            <div key={m.name} className="flex items-center justify-between text-[11px]" title={m.detail}>
              <span className="text-neutral-500">{MICRO_LABEL[m.name] ?? m.name}</span>
              <span
                className={`font-mono tabular-nums ${m.conviction < 0.99 ? 'text-amber-400' : 'text-neutral-600'}`}
              >
                {m.direction}
                {m.conviction < 0.99 && ` ×${m.conviction.toFixed(2)}`}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-2.5 text-[10px] leading-relaxed text-neutral-600">
          Micro never votes on direction here. Thirteen factors (Roll, Corwin-Schultz, Amihud, Kyle-lambda,
          intrabar pressure) were tested on 128k 15m bars with a three-way split: the best decays 59.7% &rarr;
          55.8% &rarr; 50.8%, and every combination lands at 50&ndash;51% against a long-only baseline of
          51.5&ndash;56.2%. Gold drifts, so a hand-weighted micro vote is worse than holding. The factors are
          fed to the model instead, where a tree can look for conditional structure.
        </p>
      </div>
    </div>
  )
}

function LevelsGrid({
  levels,
  action,
}: {
  levels: import('../lib/api').GoldLevels | undefined
  action: 'BUY' | 'SELL' | 'HOLD'
}) {
  if (!levels || levels.breakout_above == null || levels.breakdown_below == null) return null
  const isHold = action === 'HOLD'
  const confirm = isHold ? 'Breaks up above' : action === 'SELL' ? 'Confirms below' : 'Confirms above'
  const invalid = isHold ? 'Breaks down below' : action === 'SELL' ? 'Breaks above' : 'Breaks below'

  const cells = [
    { label: confirm, value: levels.breakout_above, pct: levels.distance_to_breakout_pct, color: 'text-emerald-400' },
    { label: invalid, value: levels.breakdown_below, pct: levels.distance_to_breakdown_pct, color: 'text-red-400' },
    { label: 'Range high', value: levels.resistance, pct: null, color: 'text-neutral-300' },
    { label: 'Range low', value: levels.support, pct: null, color: 'text-neutral-300' },
  ]

  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {cells.map((c) => (
          <div key={c.label} className="rounded-md border border-neutral-800/70 bg-neutral-950/50 p-2.5">
            <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-600">{c.label}</div>
            <div className={`mt-1 font-mono text-sm tabular-nums ${c.color}`}>{money(c.value)}</div>
            {c.pct != null && <div className="font-mono text-[10px] tabular-nums text-neutral-600">{signedPct(c.pct)}</div>}
          </div>
        ))}
      </div>
      <div className="mt-2.5 text-[10px] text-neutral-600">
        Range over the last {levels.range_window_bars} bars
        {levels.atr != null && ` · ATR ${money(levels.atr)}`}
        {levels.invalidation != null && ` · ATR invalidation ${money(levels.invalidation)}`}
      </div>
    </>
  )
}

// ------------------------------------------------------------------- HUD

function HorizonChip({
  horizon,
  prediction,
  active,
  onSelect,
}: {
  horizon: string
  prediction: GoldPrediction | undefined
  active: boolean
  onSelect: () => void
}) {
  const action = (prediction?.action ?? 'HOLD') as 'BUY' | 'SELL' | 'HOLD'
  const tone = ACTION_TONE[action]
  const trade = prediction?.trade
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`flex-1 rounded-md border px-3 py-2.5 text-left transition-colors ${
        active
          ? 'border-amber-800/70 bg-amber-950/20'
          : 'border-neutral-800/70 bg-neutral-950/40 hover:border-neutral-700'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${tone.dot}`} />
        <span className={`text-xs font-medium ${active ? 'text-neutral-100' : 'text-neutral-400'}`}>
          {HORIZON_LABEL[horizon] ?? horizon}
        </span>
        <span className={`ml-auto font-mono text-[10px] font-bold tracking-wide ${tone.text}`}>{action}</span>
      </div>
      <div className="mt-1 flex items-center gap-2 text-[10px] text-neutral-600">
        <span>{HORIZON_HINT[horizon]}</span>
        {trade && trade.risk_reward != null && (
          <span className={`ml-auto font-mono tabular-nums ${trade.viable ? 'text-emerald-500' : 'text-amber-500'}`}>
            {trade.risk_reward.toFixed(2)}x
          </span>
        )}
      </div>
    </button>
  )
}

function TradePlan({ trade }: { trade: GoldTradePlan | null | undefined }) {
  if (!trade) {
    return <div className="text-[11px] text-neutral-600">No trade plan for this horizon yet.</div>
  }

  if (trade.action === 'HOLD') {
    return (
      <div>
        <div className="mb-3 flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Trade plan</span>
          <span className="rounded bg-neutral-800/70 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-neutral-400">
            no position
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-md border border-neutral-800/70 bg-neutral-950/50 p-3">
            <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-600">Would go long above</div>
            <div className="mt-1 font-mono text-sm tabular-nums text-emerald-400/80">{money(trade.trigger_long_above)}</div>
          </div>
          <div className="rounded-md border border-neutral-800/70 bg-neutral-950/50 p-3">
            <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-600">Would go short below</div>
            <div className="mt-1 font-mono text-sm tabular-nums text-red-400/80">{money(trade.trigger_short_below)}</div>
          </div>
        </div>
        <p className="mt-2.5 text-[10px] text-neutral-600">{trade.note}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Trade plan</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider ${
            trade.viable ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
          }`}
        >
          {trade.viable ? 'takeable' : 'skip'}
        </span>
        {trade.entry_type === 'limit' && (
          <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-amber-500/90">
            wait for limit
          </span>
        )}
        <span className="ml-auto flex items-baseline gap-1.5">
          <span className="text-[9px] uppercase tracking-wider text-neutral-600">R:R</span>
          <span
            className={`font-mono text-lg tabular-nums ${trade.viable ? 'text-emerald-400' : 'text-amber-400'}`}
          >
            {trade.risk_reward == null ? '--' : `${trade.risk_reward.toFixed(2)}x`}
          </span>
          {/* What chasing costs, kept beside what waiting pays. */}
          {trade.entry_type === 'limit' && trade.risk_reward_at_market != null && (
            <span className="font-mono text-[10px] tabular-nums text-neutral-600 line-through">
              {trade.risk_reward_at_market.toFixed(2)}x
            </span>
          )}
        </span>
      </div>

      <PriceLadder trade={trade} />

      {/* Everything below is measured from the CURRENT price, because that is
          where the reader is standing -- how far to the entry, how much of the
          target is still on the table, how far the stop sits from here. */}
      {!trade.model_conflict && trade.target_remaining_usd != null && (
        <div className="mt-3 flex items-baseline justify-between rounded-md border border-neutral-800/70 bg-neutral-950/60 px-3 py-2">
          <span className="text-[10px] uppercase tracking-[0.12em] text-neutral-500">Target left from here</span>
          <span className="flex items-baseline gap-2">
            <span className="font-mono text-base tabular-nums text-emerald-400">
              {money(trade.target_remaining_usd)}
            </span>
            <span className="font-mono text-[11px] tabular-nums text-neutral-500">
              {signedPct(trade.target_remaining_pct)}
            </span>
          </span>
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[10px] text-neutral-600">
        {trade.entry_type === 'limit' && (
          <span>
            To entry{' '}
            <span className="font-mono tabular-nums text-amber-400">{signedPct(trade.entry_distance_pct)}</span>
            {trade.entry_distance_atr != null && (
              <span className="font-mono tabular-nums"> · {trade.entry_distance_atr.toFixed(1)} ATR</span>
            )}
            {trade.entry_reachable === false && <span className="text-amber-500"> · may not trade</span>}
          </span>
        )}
        <span>
          Stop <span className="font-mono tabular-nums text-neutral-400">{signedPct(trade.stop_distance_pct == null ? null : -trade.stop_distance_pct)}</span>
          {trade.stop_atr_multiple != null && (
            <span className="font-mono tabular-nums"> · {trade.stop_atr_multiple.toFixed(1)} ATR</span>
          )}
        </span>
        {trade.structural_target != null && (
          <span>
            Structure <span className="font-mono tabular-nums text-neutral-500">{money(trade.structural_target)}</span>
            {trade.structural_risk_reward != null && (
              <span className="font-mono tabular-nums text-neutral-600"> ({trade.structural_risk_reward.toFixed(2)}x, not the target)</span>
            )}
          </span>
        )}
        {trade.size_multiplier != null && trade.size_multiplier < 0.99 && (
          <span className="text-amber-500">Size &times;{trade.size_multiplier.toFixed(2)}</span>
        )}
      </div>

      <p className={`mt-2 text-[10px] leading-relaxed ${trade.viable ? 'text-neutral-600' : 'text-amber-500/85'}`}>
        {trade.note}
      </p>
    </div>
  )
}

function HorizonConsole({
  prediction,
  now,
  crossAsset,
}: {
  prediction: GoldPrediction | undefined
  now: number
  crossAsset: import('../lib/api').GoldCrossAsset | undefined
}) {
  const action = (prediction?.action ?? 'HOLD') as 'BUY' | 'SELL' | 'HOLD'
  const tone = ACTION_TONE[action]
  const risk = RISK_STYLE[prediction?.risk_level ?? 'very_high']
  const tier = prediction?.score != null ? signalTier(prediction.score) : null
  const confirmation = cryptoConfirmation(action, crossAsset)

  if (!prediction) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/40 p-8 text-center text-sm text-neutral-600">
        No model trained for this horizon yet.
      </div>
    )
  }

  return (
    <div className={`rounded-lg border bg-neutral-950/60 ${tone.ring} ${tone.glow}`}>
      <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,240px)_minmax(0,1fr)]">
        {/* --- dial + verdict --- */}
        <div className="flex flex-col items-center justify-start gap-3 border-neutral-800/60 lg:border-r lg:pr-6">
          <SignalDial score={prediction.score} action={action} />
          <div className={`font-mono text-2xl font-bold tracking-[0.14em] ${tone.text}`}>{action}</div>
          {tier && <div className="text-[11px] text-neutral-500">{tier.label}</div>}
          <div className={`rounded-full border px-2.5 py-0.5 text-[10px] ${risk.className}`}>{risk.label}</div>
          {prediction.risk_reason && (
            <p className="text-center text-[10px] leading-snug text-neutral-600">{prediction.risk_reason}</p>
          )}
          <div className="w-full pt-1">
            <AccuracyStrip
              accuracy={prediction.mean_directional_accuracy}
              samples={prediction.n_samples}
              baseline={prediction.long_only_baseline}
            />
          </div>
        </div>

        {/* --- plan + evidence --- */}
        <div className="space-y-5">
          <TradePlan trade={prediction.trade} />

          <div className="border-t border-neutral-800/60 pt-4">
            <div className="mb-2.5 text-[10px] uppercase tracking-[0.16em] text-neutral-500">Where the call came from</div>
            <EvidenceStrip blend={prediction.blend} />
          </div>

          <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 border-t border-neutral-800/60 pt-3 text-[10px] text-neutral-600">
            <span>
              Valid for <span className="font-mono tabular-nums text-neutral-400">{countdownLabel(prediction.valid_until, now)}</span>
            </span>
            {prediction.expected_move_pct != null && (
              <span>
                Model expects{' '}
                <span
                  className={`font-mono tabular-nums ${
                    prediction.expected_move_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {signedPct(prediction.expected_move_pct)}
                </span>
              </span>
            )}
            {prediction.mean_information_coefficient != null && (
              <span>
                IC <span className="font-mono tabular-nums text-neutral-400">{prediction.mean_information_coefficient.toFixed(3)}</span>
              </span>
            )}
            {confirmation && (
              <span>
                Crypto corroboration{' '}
                <span className="font-mono tabular-nums text-neutral-400">
                  {confirmation.agree}/{confirmation.total}
                </span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ page

export default function Gold() {
  const now = useNowTick(1000)
  const [selected, setSelected] = useState('intraday')

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
  const flash = useFlashOnChange(snapshot?.price ?? null)
  const prediction = snapshot?.predictions?.[selected]

  const indicatorCount = useMemo(
    () => Object.keys(snapshot?.indicators ?? {}).length,
    [snapshot?.indicators]
  )

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <style>{`
        @keyframes gold-sweep { 0% { transform: translateX(-110%); } 100% { transform: translateX(420%); } }
        .gold-sweep { animation: gold-sweep 3.4s cubic-bezier(.4,0,.6,1) infinite; }
        @keyframes gold-up { 0% { color: #4bb98a; } 100% { color: inherit; } }
        @keyframes gold-down { 0% { color: #e2645f; } 100% { color: inherit; } }
        .flash-up { animation: gold-up 800ms ease-out; }
        .flash-down { animation: gold-down 800ms ease-out; }
        @media (prefers-reduced-motion: reduce) {
          .gold-sweep, .flash-up, .flash-down { animation: none; }
        }
      `}</style>

      {/* ---- masthead ---- */}
      <div className="relative overflow-hidden rounded-lg border border-neutral-800 bg-gradient-to-b from-neutral-900/80 to-neutral-950">
        <div className="absolute inset-x-0 top-0 h-px overflow-hidden bg-neutral-800">
          <div className="gold-sweep h-px w-1/4 bg-gradient-to-r from-transparent via-amber-400 to-transparent" />
        </div>
        <div className="flex flex-wrap items-end justify-between gap-4 p-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-amber-500" />
              </span>
              <h1 className="text-sm font-medium tracking-wide text-neutral-100">Gold Research Engine</h1>
              <span className="text-[10px] uppercase tracking-[0.16em] text-neutral-600">PAXG</span>
              <span className={`rounded-full border px-2 py-0.5 text-[10px] ${regime.className}`}>{regime.label}</span>
            </div>
            <p className="mt-1.5 max-w-md text-[11px] leading-relaxed text-neutral-600">
              Research only &mdash; not paper-traded, no positions or PnL. Three independently trained horizon
              models over a dedicated feature bank, blended 50/50 with macro factors.
            </p>
          </div>
          <div className="text-right">
            <div
              className={`font-mono text-4xl tabular-nums text-neutral-50 ${
                flash === 'up' ? 'flash-up' : flash === 'down' ? 'flash-down' : ''
              }`}
            >
              {money(snapshot?.price)}
            </div>
            <div className="mt-0.5 text-[10px] text-neutral-600">{secondsAgoLabel(snapshot?.as_of, now)}</div>
          </div>
        </div>
      </div>

      {!snapshot?.ready && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-950/40 p-10 text-center text-sm text-neutral-600">
          Waiting for gold data to backfill and the first research tick to run (every ~3 minutes)&hellip;
        </div>
      )}

      {snapshot?.ready && (
        <>
          <div className="flex flex-col gap-2 sm:flex-row">
            {HORIZON_ORDER.map((h) => (
              <HorizonChip
                key={h}
                horizon={h}
                prediction={snapshot.predictions[h]}
                active={selected === h}
                onSelect={() => setSelected(h)}
              />
            ))}
          </div>

          <HorizonConsole
            prediction={prediction}
            now={now}
            crossAsset={snapshot.cross_asset}
          />

          <Section title="Levels" hint="range, confirmation and invalidation">
            <LevelsGrid levels={prediction?.levels} action={(prediction?.action ?? 'HOLD') as 'BUY' | 'SELL' | 'HOLD'} />
          </Section>

          {prediction?.blend && (
            <Section title="Factor detail" hint="what each macro and micro input is reading">
              <FactorTable blend={prediction.blend} />
              <p className="mt-4 border-t border-neutral-800/60 pt-3 text-[10px] leading-relaxed text-neutral-600">
                <span className="text-neutral-500">Weights were tested, not assumed.</span> Sweeping the
                model/macro split from 0 to 100% on out-of-fold predictions gains{' '}
                <span className="font-mono text-neutral-400">+0.4</span> points out of sample against a{' '}
                <span className="font-mono text-neutral-400">&plusmn;3.6</span> point noise band, and tilting
                the macro mix toward the dollar gains <span className="font-mono text-neutral-400">0.0</span>.
                Tightening a conviction filter looked best in-sample and inverted out of sample (56.0% &rarr;
                46.6%). The 50/50 setting stands because nothing measurably beat it.
              </p>
            </Section>
          )}

          <Section title="Price" hint="15m" defaultOpen>
            <CandleChart candles={candlesQuery.data ?? []} />
          </Section>

          <Section title="Calendar & events" hint="macro releases, seasonality, tithis">
            <EventsBody events={snapshot.events} />
          </Section>

          <Section title="Crypto correlation" hint="context only, never a forecasting input">
            <CrossAssetBody crossAsset={snapshot.cross_asset} />
          </Section>

          <Section title="Deep analysis engine" hint={`4-stage cascade over ${indicatorCount} indicators`}>
            <DeepAnalysisBody indicators={snapshot.indicators} />
          </Section>
        </>
      )}
    </div>
  )
}

// --------------------------------------------------------- section bodies

function EventsBody({ events }: { events: import('../lib/api').GoldEvents | undefined }) {
  if (!events) return null
  const { seasonality, gold_buying_tithis, next_macro_event } = events
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {next_macro_event.event && (
        <div className="rounded-md border border-amber-900/50 bg-amber-950/20 p-3">
          <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-500">Next US macro event</div>
          <div className="mt-1 text-sm font-semibold text-amber-400">
            {next_macro_event.event} in {formatHoursAway(next_macro_event.hours_away)}
          </div>
          <div className="mt-1 text-[10px] leading-snug text-neutral-600">
            {MACRO_EVENT_HINT[next_macro_event.event]}
          </div>
        </div>
      )}
      <div className="rounded-md border border-neutral-800/70 p-3">
        <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-500">Documented seasonality</div>
        <div className={`mt-1 text-sm font-semibold ${SEASONALITY_STYLE[seasonality.effect]}`}>
          {seasonality.month}: {seasonality.effect}
        </div>
        <div className="mt-1 text-[10px] leading-snug text-neutral-600">{seasonality.note}</div>
        <div className="mt-2 text-[9px] italic text-neutral-700">{seasonality.source}</div>
      </div>
      {(['shukla_tritiya', 'krishna_trayodashi'] as const).map((key) => {
        const t = gold_buying_tithis[key]
        return (
          <div key={key} className="rounded-md border border-neutral-800/70 p-3">
            <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-500">{t.label}</div>
            <div className="mt-1 text-sm font-semibold text-amber-400">{formatDaysAway(t.days_away)}</div>
            <div className="mt-1 text-[10px] text-neutral-600">{t.date ?? '--'}</div>
          </div>
        )
      })}
    </div>
  )
}

function CrossAssetBody({ crossAsset }: { crossAsset: import('../lib/api').GoldCrossAsset | undefined }) {
  if (!crossAsset?.ready) return <div className="text-[11px] text-neutral-600">Not enough overlapping data yet.</div>
  return (
    <>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {Object.entries(crossAsset.assets).map(([symbol, asset]) => {
          const style = CORRELATION_LABEL[asset.interpretation] ?? CORRELATION_LABEL.unknown
          return (
            <div key={symbol} className="rounded-md border border-neutral-800/70 p-3">
              <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-500">{symbol}</div>
              <div className={`mt-1 text-sm font-semibold ${style.className}`}>{style.label}</div>
              <div className="mt-1 font-mono text-[10px] tabular-nums text-neutral-600">
                r = {asset.correlation != null ? asset.correlation.toFixed(2) : '--'}
              </div>
            </div>
          )
        })}
      </div>
      <p className="mt-3 text-[10px] leading-relaxed text-neutral-600">
        Rolling {crossAsset.window_bars}-bar ({crossAsset.resolution}) correlation. A real, strongly significant
        SAME-INSTANT relationship &mdash; both react to shared risk-sentiment and liquidity conditions &mdash; but with
        zero exploitable lead time in either direction (checked at lags from 15 minutes to 4 hours). Context for
        reading the current move, never fed to the models.
      </p>
    </>
  )
}

function DeepAnalysisBody({ indicators }: { indicators: Record<string, number> }) {
  const composite = indicators['deep_composite_score']
  const tier = composite != null ? signalTier(composite) : null
  return (
    <>
      <div className="space-y-2.5">
        {DEEP_STAGES.map((stage) => {
          const value = indicators[stage.key]
          return (
            <div key={stage.key} className="flex items-center gap-3" title={stage.hint}>
              <div className="w-40 shrink-0 text-[11px] text-neutral-400">{stage.title}</div>
              <div className="flex-1">
                {stage.kind === 'signed' ? <SignedBar value={value ?? 0} /> : <ConfidenceBar value={value ?? 0} />}
              </div>
              <div className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-neutral-400">
                {value != null ? value.toFixed(2) : '--'}
              </div>
            </div>
          )
        })}
      </div>
      <div className="mt-4 flex items-center justify-between border-t border-neutral-800/60 pt-3">
        <div>
          <div className="text-[11px] text-neutral-400">Composite score</div>
          <div className="mt-0.5 text-[10px] text-neutral-600">
            regime confidence &times; (trend/reversion blend) + astrology tilt
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xl tabular-nums text-neutral-100">
            {composite != null ? `${composite >= 0 ? '+' : ''}${composite.toFixed(3)}` : '--'}
          </span>
          {tier && <span className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${tier.badgeClass}`}>{tier.label}</span>}
        </div>
      </div>
    </>
  )
}
