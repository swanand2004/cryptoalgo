import { useEffect, useMemo, useState } from 'react'
import { CoinCard, TradeTable } from './Analyst'
import { useQuery } from '@tanstack/react-query'
import { api, type GoldTrade, type SignalBoard } from '../lib/api'
import { CandleChart, type ChartMarker, type PriceLine } from '../components/CandleChart'
import { formatISTShort } from '../lib/time'
import { ErrorState } from '../components/ui'
import { useIsPhone } from '../lib/hooks'

// Gold page: the current call per timeframe with its trade levels, a chart
// with those levels drawn on it, and the support/resistance map. How the call
// is computed lives in app/gold/signal_board.py, not on the page.

type Timeframe = 'hourly' | 'daily'
type Board = SignalBoard | null | undefined
type Trade = NonNullable<SignalBoard['trade']>

const TIMEFRAMES: Record<Timeframe, { label: string; sub: string; resolution: string; candles: number; barSeconds: number }> = {
  hourly: { label: 'Intraday', sub: '1H · next few hours', resolution: '1h', candles: 240, barSeconds: 3600 },
  daily: { label: 'Swing', sub: '1D · next few days', resolution: '1d', candles: 100, barSeconds: 86_400 },
}

const DIRECTION_TEXT = { BUY: 'text-emerald-400', SELL: 'text-red-400', NEUTRAL: 'text-neutral-300' } as const
const DIRECTION_BAR = { BUY: 'bg-emerald-500', SELL: 'bg-red-500', NEUTRAL: 'bg-neutral-500' } as const

const STATUS_STYLE: Record<Trade['status'], string> = {
  ACTIVE: 'border-sky-800/70 bg-sky-950/40 text-sky-300',
  'TP HIT': 'border-emerald-700 bg-emerald-900/40 text-emerald-200',
  'STOPPED OUT': 'border-red-900/70 bg-red-950/40 text-red-300',
}

// How each finished trade ended, for the card and the trade logs.
const RESULT_STYLE: Record<string, { label: string; cls: string; marker: string }> = {
  TP: { label: 'TP1 hit', cls: 'border-emerald-800/70 bg-emerald-950/50 text-emerald-300', marker: '#34d399' },
  SL: { label: 'SL hit', cls: 'border-red-900/70 bg-red-950/50 text-red-300', marker: '#f87171' },
}

function statusLabel(status: Trade['status']): string {
  return status === 'TP HIT' ? 'TP1 HIT' : status
}

const EVENT_NAME: Record<string, string> = { FOMC: 'Fed rate decision', CPI: 'US inflation (CPI)', NFP: 'US jobs report' }

const money = (v: number) => v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const signed = (v: number, digits = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

function ago(iso: string | null | undefined, now: number): string {
  if (!iso) return ''
  const mins = Math.max(0, Math.round((now - new Date(iso).getTime()) / 60_000))
  if (mins < 60) return `${mins}m ago`
  const hours = Math.round(mins / 60)
  if (hours < 48) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

function inHours(hours: number | null): string {
  if (hours === null) return ''
  if (hours < 1) return 'within the hour'
  if (hours < 48) return `in ${Math.round(hours)}h`
  return `in ${Math.round(hours / 24)} days`
}

// ------------------------------------------------------------------ pieces

function Panel({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="min-w-0 rounded-xl border border-neutral-800 bg-neutral-900/50">
      <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-neutral-800/70 px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">{title}</h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

function Header({ price, change, asOf, now, event }: {
  price: number | null
  change: number | null
  asOf: string | null | undefined
  now: number
  event: { event: string | null; hours_away: number | null; at: string | null } | undefined
}) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-neutral-800 bg-gradient-to-br from-amber-950/40 via-neutral-900 to-neutral-950 p-5 sm:p-6">
      <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-amber-500/10 blur-3xl" />
      <div className="relative flex flex-wrap items-end justify-between gap-5">
        <div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-60 motion-reduce:animate-none" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
            </span>
            <span className="text-xs font-medium uppercase tracking-[0.22em] text-amber-300">Gold</span>
            <span className="text-xs text-neutral-400">XAU/USD · Delta XAUT</span>
          </div>
          <div className="mt-2 font-mono text-4xl tabular-nums text-neutral-50 sm:text-5xl">
            {price !== null ? `$${money(price)}` : '--'}
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-3 text-sm">
            {change !== null && (
              <span
                className={`rounded-md px-2 py-0.5 font-mono ${
                  change >= 0 ? 'bg-emerald-950/60 text-emerald-300' : 'bg-red-950/60 text-red-300'
                }`}
              >
                {signed(change)}% 24h
              </span>
            )}
            {asOf && <span className="text-xs text-neutral-400">updated {ago(asOf, now)}</span>}
          </div>
        </div>
        {event?.event && (
          <div className="rounded-lg border border-amber-900/50 bg-neutral-950/50 px-4 py-2.5 sm:text-right">
            <div className="text-[10px] uppercase tracking-[0.16em] text-neutral-400">Next market event</div>
            <div className="mt-0.5 font-medium text-amber-300">{EVENT_NAME[event.event] ?? event.event}</div>
            <div className="text-xs text-neutral-400">
              {inHours(event.hours_away)}
              {event.at && <span className="text-neutral-400"> · {formatISTShort(event.at)}</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function LevelBox({ label, value, price, tone, hit }: {
  label: string
  value: number
  price: number
  tone: 'neutral' | 'red' | 'green'
  hit?: boolean
}) {
  const color = tone === 'red' ? 'text-red-300' : tone === 'green' ? 'text-emerald-300' : 'text-neutral-100'
  const distance = (value / price - 1) * 100
  return (
    <div
      className={`min-w-0 rounded-lg border px-2 py-2 sm:px-3 ${
        hit ? 'border-emerald-700/70 bg-emerald-950/30' : 'border-neutral-800 bg-neutral-950/60'
      }`}
    >
      <div className="flex items-center justify-between gap-1 text-[10px] uppercase tracking-wide text-neutral-400">
        {label}
        {hit && <span className="normal-case tracking-normal text-emerald-400">&#10003; hit</span>}
      </div>
      <div className={`mt-0.5 truncate font-mono text-[13px] tabular-nums sm:text-[15px] ${color}`}>{money(value)}</div>
      <div className="font-mono text-[10px] text-neutral-400">{signed(distance)}%</div>
    </div>
  )
}

function SignalCard({ timeframe, board, livePrice, active, now, onSelect }: {
  timeframe: Timeframe
  board: Board
  livePrice: number | null
  active: boolean
  now: number
  onSelect: () => void
}) {
  const meta = TIMEFRAMES[timeframe]
  const frame = `min-w-0 rounded-xl border p-4 text-left transition-colors sm:p-5 ${
    active
      ? 'border-amber-500/50 bg-neutral-900 ring-1 ring-amber-500/20'
      : 'border-neutral-800 bg-neutral-900/50 hover:border-neutral-700'
  }`
  if (!board) {
    return (
      <div className={frame}>
        <div className="text-[11px] uppercase tracking-[0.16em] text-neutral-500">{meta.label}</div>
        <div className="mt-6 h-10 w-32 animate-pulse rounded bg-neutral-800" />
        <div className="mt-6 h-16 animate-pulse rounded bg-neutral-800/60" />
      </div>
    )
  }
  const c = board.consensus
  const t = board.trade
  // The current signal's trade already hit its target/stop: it is off the card, with
  // its result shown until the next signal. ISO UTC strings compare in order.
  const finished =
    !t && c.direction !== 'NEUTRAL' && board.last_trade && board.last_trade.entry_time >= c.since
      ? board.last_trade
      : null
  const price = livePrice ?? board.price
  // Closed trades keep the P&L they closed at; open ones mark to the live price.
  // The trade's own side: it can outlive (and differ from) the current signal.
  const side = t?.side === 'SELL' ? -1 : 1
  const open = t !== null && t.closed_time === null
  const pnl = t ? (open ? side * (price / t.entry - 1) * 100 : t.pnl_pct) : 0
  const rMultiple = t ? (open ? (side * (price - t.entry)) / Math.abs(t.entry - t.stop_loss) : t.r_multiple) : 0
  const strength = Math.round(Math.min(1, Math.abs(c.score)) * 100)
  return (
    <button type="button" onClick={onSelect} aria-pressed={active} className={frame}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-[0.16em] text-neutral-300">{meta.label}</div>
          <div className="text-[11px] text-neutral-400">{meta.sub}</div>
        </div>
        {t && (
          <span className={`rounded-full border px-2.5 py-0.5 text-[10px] font-medium tracking-wide ${STATUS_STYLE[t.status]}`}>
            {statusLabel(t.status)}
          </span>
        )}
      </div>

      <div className="mt-3 flex items-end justify-between gap-4">
        <div className={`text-4xl font-semibold tracking-tight ${DIRECTION_TEXT[c.direction]}`}>
          {c.direction === 'NEUTRAL' ? 'WAIT' : c.direction}
        </div>
        {t && (
          <div className="text-right">
            <div className={`font-mono text-2xl tabular-nums ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {signed(pnl)}%
            </div>
            <div className="font-mono text-[11px] text-neutral-400">{signed(rMultiple)}R</div>
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center gap-3">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-neutral-800">
          <div className={`h-full rounded-full ${DIRECTION_BAR[c.direction]}`} style={{ width: `${strength}%` }} />
        </div>
        <span className="text-[11px] text-neutral-400">strength {strength}%</span>
      </div>

      <div className="mt-3 text-[12px] text-neutral-400">
        Signal <span className="font-mono text-neutral-200">{formatISTShort(c.since)}</span>
        <span className="text-neutral-400"> · {ago(c.since, now)}</span>
      </div>

      {t ? (
        <div className="mt-4 grid grid-cols-3 gap-2">
          <LevelBox label="Entry" value={t.entry} price={price} tone="neutral" />
          <LevelBox label="Stop loss" value={t.stop_loss} price={price} tone="red" />
          <LevelBox label="TP1" value={t.take_profit} price={price} tone="green" hit={t.status === 'TP HIT'} />
        </div>
      ) : (
        <>
          {finished && (
            <div className={`mt-4 rounded-lg border px-3 py-2 text-[12px] ${RESULT_STYLE[finished.result ?? 'SL'].cls}`}>
              This signal&rsquo;s trade is closed &mdash; {RESULT_STYLE[finished.result ?? 'SL'].label} at{' '}
              <span className="font-mono">{money(finished.exit_price ?? 0)}</span>{' '}
              <span className="font-mono">({signed(finished.pnl_pct)}%)</span>, {formatISTShort(finished.exit_time)}.
              {!board.plan && <span className="block text-neutral-400">Waiting for the next signal.</span>}
            </div>
          )}
          {board.plan && (
            <>
              <div className="mt-4 grid grid-cols-3 gap-2">
                <LevelBox label="Entry" value={board.plan.entry} price={price} tone="neutral" />
                <LevelBox label="Stop loss" value={board.plan.stop_loss} price={price} tone="red" />
                <LevelBox label="TP1" value={board.plan.take_profit} price={price} tone="green" />
              </div>
              <div className="mt-2 text-[11px] text-neutral-400">
                {board.plan.side} signal active, no open trade &mdash; levels for a trade entered now.
              </div>
            </>
          )}
          {board.triggers && (
            <div className="mt-4 grid grid-cols-2 gap-2">
              <LevelBox label="Buy above" value={board.triggers.buy_above} price={price} tone="green" />
              <LevelBox label="Sell below" value={board.triggers.sell_below} price={price} tone="red" />
            </div>
          )}
        </>
      )}
    </button>
  )
}

function LevelLadder({ board, price }: { board: SignalBoard; price: number }) {
  // Levels from different sources that land on the same price read as one.
  const merged: { price: number; labels: string[]; kind: string }[] = []
  for (const lv of board.levels) {
    const prev = merged[merged.length - 1]
    if (prev && Math.abs(prev.price / lv.price - 1) < 0.0005) prev.labels.push(lv.label)
    else merged.push({ price: lv.price, labels: [lv.label], kind: lv.price > price ? 'resistance' : 'support' })
  }
  const above = merged.filter((l) => l.kind === 'resistance').slice(-6)
  const below = merged.filter((l) => l.kind === 'support').slice(0, 6)
  const row = (l: (typeof merged)[number], tone: string) => (
    <div key={l.price} className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="min-w-0 truncate text-[12px] text-neutral-400" title={l.labels.join(' · ')}>
        {l.labels.join(' · ')}
      </span>
      <span className="shrink-0 text-right">
        <span className={`font-mono text-[13px] tabular-nums ${tone}`}>{money(l.price)}</span>
        <span className="ml-2 inline-block w-14 font-mono text-[10px] text-neutral-400">
          {signed((l.price / price - 1) * 100)}%
        </span>
      </span>
    </div>
  )
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-[0.16em] text-red-400/70">Resistance</div>
      <div className="divide-y divide-neutral-800/60">{above.map((l) => row(l, 'text-red-300'))}</div>
      <div className="my-2 flex items-center justify-between rounded-md border border-amber-800/60 bg-amber-950/30 px-2.5 py-1.5">
        <span className="text-[11px] uppercase tracking-[0.14em] text-amber-300">Price</span>
        <span className="font-mono text-[13px] text-amber-200">{money(price)}</span>
      </div>
      <div className="divide-y divide-neutral-800/60">{below.map((l) => row(l, 'text-emerald-300'))}</div>
      <div className="mt-1 text-[10px] uppercase tracking-[0.16em] text-emerald-400/70">Support</div>
    </div>
  )
}

const SUMMARY_TONE: Record<string, string> = {
  'Strong Buy': 'text-emerald-300',
  Buy: 'text-emerald-400',
  Neutral: 'text-neutral-300',
  Sell: 'text-red-400',
  'Strong Sell': 'text-red-300',
}
const FAMILY_NAME: Record<string, string> = { Trend: 'Trend', Oscillator: 'Oscillators', Gann: 'Gann' }

function SummaryTiles({ board, daily }: { board: SignalBoard; daily: Board }) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {board.families.map((f) => (
        <div key={f.family} className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
          <div className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">{FAMILY_NAME[f.family] ?? f.family}</div>
          <div className={`mt-1 text-lg font-semibold ${SUMMARY_TONE[f.label]}`}>{f.label}</div>
          <div className="relative mt-2 h-1 rounded-full bg-neutral-800">
            <div className="absolute inset-y-0 left-1/2 w-px bg-neutral-600" />
            <div
              className={`absolute inset-y-0 rounded-full ${f.score >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
              style={f.score >= 0 ? { left: '50%', width: `${f.score * 50}%` } : { right: '50%', width: `${-f.score * 50}%` }}
            />
          </div>
        </div>
      ))}
      <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 p-4">
        <div className="text-[10px] uppercase tracking-[0.16em] text-neutral-500">Gann dates</div>
        {(daily?.gann.time_cycles ?? []).slice(0, 3).map((c) => (
          <div key={`${c.from}${c.bars}`} className="mt-1 flex items-baseline justify-between gap-2 text-[12px]">
            <span className="font-mono text-neutral-200">
              {new Date(c.due).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', timeZone: 'Asia/Kolkata' })}
            </span>
            <span className="text-neutral-500">
              {c.bars}d from {c.from === 'top' ? 'high' : 'low'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// The board reports bar CLOSE times; the chart is keyed by bar OPEN times.
function barOpen(iso: string, barSeconds: number): number {
  return Math.round(new Date(iso).getTime() / 1000) - barSeconds
}

function tradeLines(board: Board, barSeconds: number): PriceLine[] {
  if (!board) return []
  const t = board.trade
  if (t) {
    const from = barOpen(t.entry_time, barSeconds)
    return [
      { price: t.entry, color: '#a3a3a3', title: 'Entry', dashed: true, from },
      { price: t.stop_loss, color: '#f87171', title: 'SL', from },
      { price: t.take_profit, color: '#34d399', title: 'TP1', from },
    ]
  }
  if (board.plan) {
    const from = barOpen(board.bar_close_time, barSeconds)
    return [
      { price: board.plan.entry, color: '#a3a3a3', title: 'Entry', dashed: true, from },
      { price: board.plan.stop_loss, color: '#f87171', title: 'SL', dashed: true, from },
      { price: board.plan.take_profit, color: '#34d399', title: 'TP1', dashed: true, from },
    ]
  }
  if (board.triggers) {
    // Breakout levels are the last 20 bars' range, so they start 20 bars back.
    const from = barOpen(board.bar_close_time, barSeconds) - 20 * barSeconds
    return [
      { price: board.triggers.buy_above, color: '#34d399', title: 'Buy above', dashed: true, from },
      { price: board.triggers.sell_below, color: '#f87171', title: 'Sell below', dashed: true, from },
    ]
  }
  return []
}

// Entry arrows and exit dots for the open trade and recent closed ones. The
// chart drops any that fall outside its visible candles.
function tradeMarkers(board: Board, barSeconds: number): ChartMarker[] {
  if (!board) return []
  const trades = [...(board.trade ? [board.trade] : []), ...board.history.slice(0, 40)]
  const out: ChartMarker[] = []
  for (const t of trades) {
    const long = t.side === 'BUY'
    out.push({
      time: barOpen(t.entry_time, barSeconds),
      position: long ? 'belowBar' : 'aboveBar',
      shape: long ? 'arrowUp' : 'arrowDown',
      color: long ? '#34d399' : '#f87171',
      text: t.open ? t.side : '',
    })
    if (!t.open && t.exit_time && t.result) {
      const style = RESULT_STYLE[t.result]
      const lossSide = t.pnl_pct < 0
      out.push({
        time: barOpen(t.exit_time, barSeconds),
        position: long === lossSide ? 'belowBar' : 'aboveBar',
        shape: 'circle',
        color: style.marker,
        text: t.result === 'TP' ? 'TP1' : t.result,
      })
    }
  }
  return out
}

function tradeTime(iso: string | null, daily: boolean): string {
  if (!iso) return '--'
  const opts: Intl.DateTimeFormatOptions = daily
    ? { day: '2-digit', month: 'short', year: '2-digit', timeZone: 'Asia/Kolkata' }
    : { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' }
  return new Date(iso).toLocaleString('en-IN', opts)
}

function TradeRow({ t, daily }: { t: GoldTrade; daily: boolean }) {
  const style = t.result ? RESULT_STYLE[t.result] : null
  return (
    <tr className={`border-b border-neutral-900 ${t.open ? 'bg-sky-950/20' : ''}`}>
      <td className="py-1.5 pr-3 font-mono text-neutral-400">{tradeTime(t.entry_time, daily)}</td>
      <td className={`py-1.5 pr-3 font-medium ${t.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side}</td>
      <td className="py-1.5 pr-3 font-mono text-neutral-200">{money(t.entry)}</td>
      <td className="py-1.5 pr-3 font-mono text-red-300/80">{money(t.stop_loss)}</td>
      <td className="py-1.5 pr-3 font-mono text-emerald-300/80">{money(t.take_profit)}</td>
      <td className="py-1.5 pr-3 font-mono text-neutral-400">{t.open ? 'open' : tradeTime(t.exit_time, daily)}</td>
      <td className="py-1.5 pr-3 font-mono text-neutral-200">{t.exit_price !== null ? money(t.exit_price) : '--'}</td>
      <td className="py-1.5 pr-3">
        {t.open ? (
          <span className={`rounded-full border px-2 py-0.5 text-[10px] ${STATUS_STYLE[t.status]}`}>{statusLabel(t.status)}</span>
        ) : (
          style && <span className={`rounded-full border px-2 py-0.5 text-[10px] ${style.cls}`}>{style.label}</span>
        )}
      </td>
      <td className={`py-1.5 text-right font-mono ${t.pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {signed(t.pnl_pct)}%
      </td>
    </tr>
  )
}

function TradeLog({ title, sub, board, daily }: { title: string; sub: string; board: Board; daily: boolean }) {
  const st = board?.history_stats
  return (
    <Panel title={title} right={<span className="text-[11px] text-neutral-400">{sub}</span>}>
      {!board ? (
        <div className="h-48 animate-pulse rounded bg-neutral-800/50" />
      ) : (
        <>
          {st && st.trades > 0 && (
            <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-neutral-400">
              <span>
                <span className="text-neutral-200">{st.trades}</span> closed trades since {tradeTime(st.since ?? null, true)}
              </span>
              <span className="text-emerald-400">{st.tp} TP1</span>
              <span className="text-red-400">{st.sl} SL</span>
              <span>win {Math.round((st.win_rate ?? 0) * 100)}%</span>
              <span className={(st.net_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                net {signed(st.net_pct ?? 0)}%
              </span>
            </div>
          )}
          <div className="max-h-96 overflow-auto">
            <table className="w-full min-w-[760px] text-[12px]">
              <thead className="sticky top-0 bg-neutral-950">
                <tr className="border-b border-neutral-800 text-left text-[11px] uppercase tracking-wider text-neutral-400">
                  <th className="py-2 pr-3 font-medium">Entry time</th>
                  <th className="py-2 pr-3 font-medium">Side</th>
                  <th className="py-2 pr-3 font-medium">Entry</th>
                  <th className="py-2 pr-3 font-medium">SL</th>
                  <th className="py-2 pr-3 font-medium">TP1</th>
                  <th className="py-2 pr-3 font-medium">Exit time</th>
                  <th className="py-2 pr-3 font-medium">Exit</th>
                  <th className="py-2 pr-3 font-medium">Result</th>
                  <th className="py-2 text-right font-medium">P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {board.trade && <TradeRow t={board.trade} daily={daily} />}
                {board.history.map((t) => (
                  <TradeRow key={t.entry_time} t={t} daily={daily} />
                ))}
                {!board.trade && board.history.length === 0 && (
                  <tr>
                    <td colSpan={9} className="py-6 text-center text-neutral-400">No trades yet</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}

// ------------------------------------------------------------------ page

// The same AI Analyst the crypto Dashboard runs, on gold (Delta XAUTUSD):
// trend, momentum, chart patterns, Gann, levels, the dollar, real yields and the
// event calendar -- every 5 minutes, paper trades with the same dynamic exits.
function GoldAnalyst() {
  const q = useQuery({ queryKey: ['analyst_gold'], queryFn: api.analystGold, refetchInterval: 60_000 })
  const d = q.data
  const st = d?.stats
  return (
    <Panel
      title="AI Analyst · gold"
      right={<span className="text-[11px] text-neutral-400">every 5 min · paper · updated {d?.updated ? formatISTShort(d.updated) : '--'}</span>}
    >
      {q.isError && !d ? (
        <ErrorState what="the gold Analyst" error={q.error} onRetry={() => q.refetch()} />
      ) : !d ? (
        <div className="h-40 animate-pulse rounded bg-neutral-800/50" />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]">
          {d.analysis ? (
            <CoinCard a={d.analysis} best={d.analysis.decision !== 'WAIT'} />
          ) : (
            <div className="text-sm text-neutral-300">The first gold analysis runs a few minutes after the server starts.</div>
          )}
          <div className="space-y-3">
            {st && (
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-neutral-400">
                <span>
                  <span className="text-neutral-200">{st.closed}</span> closed
                </span>
                <span>win {st.win_rate != null ? `${Math.round(st.win_rate * 100)}%` : '--'}</span>
                <span>avg {st.avg_r != null ? `${st.avg_r >= 0 ? '+' : ''}${st.avg_r.toFixed(2)}R` : '--'}</span>
                <span className={st.total_r >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                  total {st.total_r >= 0 ? '+' : ''}
                  {st.total_r.toFixed(2)}R
                </span>
              </div>
            )}
            <TradeTable rows={d.open_trades} open />
            {d.closed_trades.length > 0 && <TradeTable rows={d.closed_trades.slice(0, 10)} open={false} />}
          </div>
        </div>
      )}
    </Panel>
  )
}

export default function Gold() {
  const now = useNow(30_000)
  const [timeframe, setTimeframe] = useState<Timeframe>('daily')

  const boards = useQuery({ queryKey: ['gold_signal_board'], queryFn: api.goldSignalBoard, refetchInterval: 60_000 })
  const snapshot = useQuery({ queryKey: ['gold_snapshot'], queryFn: api.goldSnapshot, refetchInterval: 30_000 })
  const meta = TIMEFRAMES[timeframe]
  const candles = useQuery({
    queryKey: ['gold_candles', meta.resolution],
    queryFn: () => api.goldCandles(meta.resolution, meta.candles),
    refetchInterval: 60_000,
  })

  const board = boards.data?.[timeframe] ?? null
  const lines = useMemo(() => tradeLines(board, meta.barSeconds), [board, meta.barSeconds])
  const markers = useMemo(() => tradeMarkers(board, meta.barSeconds), [board, meta.barSeconds])
  const price = snapshot.data?.price ?? boards.data?.hourly?.price ?? null
  const isPhone = useIsPhone()

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <Header
        price={price}
        change={boards.data?.hourly?.change_24h_pct ?? null}
        asOf={snapshot.data?.as_of ?? boards.data?.hourly?.bar_close_time}
        now={now}
        event={snapshot.data?.events?.next_macro_event}
      />

      {boards.isError && (
        <div className="rounded-xl border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-300">
          Signals are unavailable right now. {String(boards.error).slice(0, 160)}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {(['hourly', 'daily'] as const).map((tf) => (
          <SignalCard
            key={tf}
            timeframe={tf}
            board={boards.isLoading ? undefined : boards.data?.[tf]}
            livePrice={price}
            active={timeframe === tf}
            now={now}
            onSelect={() => setTimeframe(tf)}
          />
        ))}
      </div>

      <GoldAnalyst />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Panel
          title={`${meta.label} chart`}
          right={
            <div className="flex gap-3 text-[11px] text-neutral-400">
              <span>
                <span className="text-neutral-400">- -</span> entry
              </span>
              <span>
                <span className="text-red-400">&mdash;</span> stop
              </span>
              <span>
                <span className="text-emerald-400">&mdash;</span> targets
              </span>
            </div>
          }
        >
          {candles.isError ? (
            <ErrorState what="gold candles" error={candles.error} onRetry={() => candles.refetch()} />
          ) : (
            <CandleChart candles={candles.data ?? []} lines={lines} markers={markers} height={isPhone ? 300 : 420} />
          )}
        </Panel>
        <Panel title="Key levels">
          {boards.isError ? (
            <div className="text-sm text-neutral-400">Levels are unavailable while signals are down.</div>
          ) : board ? (
            <LevelLadder board={board} price={price ?? board.price} />
          ) : boards.isLoading ? (
            <div className="h-80 animate-pulse rounded bg-neutral-800/50" />
          ) : (
            <div className="text-sm text-neutral-400">No {meta.label.toLowerCase()} levels yet.</div>
          )}
        </Panel>
      </div>

      {/* Stacked, not side by side: each log needs the full width for its ten
          columns, or the exit / result / P&L -- the part people read -- scroll away. */}
      <div className="grid gap-4">
        <TradeLog title="Short-term trades" sub="1H signals · hours" board={boards.data?.hourly} daily={false} />
        <TradeLog title="Long-term trades" sub="1D signals · days" board={boards.data?.daily} daily />
      </div>

      {board && <SummaryTiles board={board} daily={boards.data?.daily} />}

      <p className="pb-2 text-center text-[11px] text-neutral-400">
        Research signals for information only &mdash; not financial advice.
      </p>
    </div>
  )
}
