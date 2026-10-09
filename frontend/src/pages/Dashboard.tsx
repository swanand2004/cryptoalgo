import { useQuery } from '@tanstack/react-query'
import { api, netPnl, type ClosedTradeRow, type Position } from '../lib/api'
import { useDashboardSocket } from '../lib/ws'
import { EquityChart } from '../components/EquityChart'
import { AnalystPanel } from '../components/AnalystPanel'
import { Empty, ErrorState, Field, Loading, Panel } from '../components/ui'
import { useIsPhone } from '../lib/hooks'
import { formatISTShort, isTodayIST } from '../lib/time'
import { ago, coin, exitReason, fmtPct, fmtPrice, fmtUsd, pnlClass } from '../lib/format'

const RECENT_CLOSED_TRADES_LIMIT = 10
// The Analyst re-runs every minute; older than this means it has stalled.
const ANALYST_STALE_MINUTES = 5

export default function Dashboard() {
  const isPhone = useIsPhone()
  const { snapshot, connected } = useDashboardSocket()
  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })
  const mode = modeQuery.data?.mode ?? 'paper'

  const equityCurveQuery = useQuery({
    queryKey: ['equity_curve', mode],
    queryFn: () => api.equityCurve(mode),
    refetchInterval: 15000,
  })
  // REST is only the fallback for first paint / if the socket drops -- once
  // connected, the WS snapshot (pushed every second) drives the positions so
  // unrealized PnL tracks live price ticks instead of a slow 10s poll.
  const positionsQuery = useQuery({
    queryKey: ['positions'],
    queryFn: api.positions,
    refetchInterval: 10000,
  })
  const positions: Position[] = snapshot?.positions ?? positionsQuery.data ?? []
  const positionsKnown = !!snapshot || positionsQuery.isSuccess

  const closedTradesQuery = useQuery({
    queryKey: ['closed_trades', mode],
    queryFn: () => api.closedTrades(mode),
    refetchInterval: 10000,
  })
  const analystQuery = useQuery({ queryKey: ['analyst'], queryFn: api.analyst, refetchInterval: 30_000 })

  // Live equity from the socket. null there means Delta's balance can't be
  // read right now; then (or before the socket connects) fall back to the
  // last recorded equity snapshot and say so.
  const liveEquity = snapshot?.equity ?? null
  const lastRecorded = equityCurveQuery.data?.at(-1)?.equity ?? null
  const equity = liveEquity ?? lastRecorded
  const equityIsStale = liveEquity == null && lastRecorded != null

  const unrealizedPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
  // Realized PnL is scoped to trades that actually CLOSED today (IST) and is
  // net of fees, so it reflects money actually kept. Open positions' own
  // realized_pnl / fees_paid have no per-close timestamp, so they are left out
  // rather than mixing in undated history.
  const todaysTrades = (closedTradesQuery.data ?? []).filter((t) => isTodayIST(t.exit_time))
  const realizedToday = todaysTrades.reduce((sum, t) => sum + netPnl(t), 0)
  const feesToday = todaysTrades.reduce((sum, t) => sum + t.fees, 0)

  const recentClosedTrades = (closedTradesQuery.data ?? []).slice(0, RECENT_CLOSED_TRADES_LIMIT)
  const prices = snapshot?.prices ?? {}

  return (
    <div className="mx-auto max-w-7xl space-y-4 sm:space-y-6">
      {/* ---- Money first ---- */}
      <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-4 sm:p-5" aria-label="Account">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-wide text-neutral-400">Equity</div>
            <div className="mt-0.5 truncate font-mono text-4xl font-semibold tabular-nums text-neutral-50">
              {equity == null ? (equityCurveQuery.isLoading && !snapshot ? '...' : '--') : fmtUsd(equity)}
            </div>
          </div>
          <ModeBadge mode={modeQuery.data?.mode ?? null} />
        </div>
        {equityIsStale && (
          <div className="mt-1 text-xs text-amber-300">
            {snapshot ? 'Delta balance unreadable right now' : 'Live feed not connected'} -- showing the last recorded value.
          </div>
        )}
        <div className="mt-4 grid grid-cols-3 gap-3 border-t border-neutral-800 pt-3">
          <Field label="Today realized">
            {!closedTradesQuery.isSuccess ? (
              <span className="text-neutral-400">--</span>
            ) : (
              <span className={pnlClass(realizedToday)}>{fmtUsd(realizedToday, { signed: true })}</span>
            )}
          </Field>
          <Field label="Open P&L">
            <span className={pnlClass(unrealizedPnl)}>{positionsKnown ? fmtUsd(unrealizedPnl, { signed: true }) : '--'}</span>
          </Field>
          <Field label="Positions" align="right">
            <span className="text-neutral-100">{positionsKnown ? positions.length : '--'}</span>
          </Field>
        </div>
        <div className="mt-2 text-[11px] text-neutral-400">
          {closedTradesQuery.isError
            ? 'Closed trades could not be loaded, so today’s realized P&L is unknown.'
            : `Today = trades closed since midnight IST, net of ${fmtUsd(feesToday)} fees (${todaysTrades.length} closed).`}
        </div>
      </section>

      {/* ---- Is the bot healthy? ---- */}
      <HealthStrip
        modeError={modeQuery.isError}
        killSwitch={modeQuery.data?.kill_switch ?? null}
        killReason={modeQuery.data?.kill_switch_reason?.message ?? null}
        connected={connected}
        systemStatus={snapshot?.system_status ?? null}
        analystUpdated={analystQuery.data?.updated ?? null}
        analystError={analystQuery.isError}
      />

      {/* ---- Open positions ---- */}
      <Panel title={`Open positions${positionsKnown ? ` (${positions.length})` : ''}`} bodyClassName={isPhone ? 'p-3' : 'p-4'}>
        {!positionsKnown ? (
          positionsQuery.isError ? (
            <ErrorState what="positions" error={positionsQuery.error} onRetry={() => positionsQuery.refetch()} />
          ) : (
            <Loading className="h-28" label="Loading positions" />
          )
        ) : positions.length === 0 ? (
          <Empty>No open positions</Empty>
        ) : isPhone ? (
          <div className="space-y-3">
            {positions.map((p) => (
              <PositionCard key={`${p.symbol}-${p.strategy}`} p={p} ltp={prices[p.symbol] ?? null} />
            ))}
          </div>
        ) : (
          <PositionsTable positions={positions} prices={prices} />
        )}
      </Panel>

      {/* ---- What the Analyst thinks ---- */}
      <AnalystPanel />

      {/* ---- Equity curve ---- */}
      <Panel title={`Equity curve · ${mode}`}>
        {equityCurveQuery.isError ? (
          <ErrorState what="the equity curve" error={equityCurveQuery.error} onRetry={() => equityCurveQuery.refetch()} />
        ) : equityCurveQuery.isLoading ? (
          <Loading className={isPhone ? 'h-[200px]' : 'h-[260px]'} label="Loading equity curve" />
        ) : equityCurveQuery.data && equityCurveQuery.data.length > 1 ? (
          <EquityChart points={equityCurveQuery.data} height={isPhone ? 200 : 260} />
        ) : (
          <Empty>Collecting equity history -- a snapshot is recorded every minute.</Empty>
        )}
      </Panel>

      {/* ---- Recent closed trades ---- */}
      <Panel title={`Recent closed trades · ${mode}`} bodyClassName={isPhone ? 'p-3' : 'p-4'}>
        {closedTradesQuery.isError ? (
          <ErrorState what="closed trades" error={closedTradesQuery.error} onRetry={() => closedTradesQuery.refetch()} />
        ) : closedTradesQuery.isLoading ? (
          <Loading className="h-28" label="Loading closed trades" />
        ) : recentClosedTrades.length === 0 ? (
          <Empty>No closed trades yet in {mode} mode</Empty>
        ) : isPhone ? (
          <ul className="divide-y divide-neutral-800">
            {recentClosedTrades.map((t) => (
              <ClosedTradeItem key={t.id} t={t} />
            ))}
          </ul>
        ) : (
          <ClosedTradesTable trades={recentClosedTrades} />
        )}
      </Panel>
    </div>
  )
}

// ------------------------------------------------------------------ health

function ModeBadge({ mode }: { mode: string | null }) {
  if (mode == null) {
    return <span className="shrink-0 rounded-full border border-neutral-700 px-2.5 py-1 text-xs font-semibold text-neutral-400">--</span>
  }
  const live = mode === 'live'
  return (
    <span
      className={`shrink-0 rounded-full border px-2.5 py-1 text-xs font-bold tracking-wide ${
        live ? 'border-red-700 bg-red-600/20 text-red-300' : 'border-sky-800 bg-sky-500/10 text-sky-300'
      }`}
    >
      {live ? 'LIVE' : 'PAPER'}
    </span>
  )
}

type Tone = 'ok' | 'warn' | 'bad' | 'idle'
const DOT: Record<Tone, string> = {
  ok: 'bg-emerald-400',
  warn: 'bg-amber-400',
  bad: 'bg-red-500',
  idle: 'bg-neutral-500',
}

function Chip({ tone, label, value }: { tone: Tone; label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-2">
      <span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${DOT[tone]}`} />
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
        <div className={`truncate text-[13px] font-medium ${tone === 'bad' ? 'text-red-300' : tone === 'warn' ? 'text-amber-300' : 'text-neutral-100'}`}>
          {value}
        </div>
      </div>
    </div>
  )
}

function HealthStrip({
  modeError,
  killSwitch,
  killReason,
  connected,
  systemStatus,
  analystUpdated,
  analystError,
}: {
  modeError: boolean
  killSwitch: boolean | null
  killReason: string | null
  connected: boolean
  systemStatus: string | null
  analystUpdated: string | null
  analystError: boolean
}) {
  const analystAgeMin = analystUpdated ? (Date.now() - new Date(analystUpdated).getTime()) / 60_000 : null
  const exchangeTone: Tone =
    systemStatus == null || systemStatus === 'unknown' ? 'idle' : systemStatus === 'maintenance' ? 'bad' : 'ok'
  return (
    <section aria-label="Bot health" className="space-y-2">
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Chip
          tone={modeError || killSwitch == null ? 'idle' : killSwitch ? 'bad' : 'ok'}
          label="Kill switch"
          value={modeError ? 'unknown' : killSwitch == null ? '...' : killSwitch ? 'ON -- halted' : 'off -- trading'}
        />
        <Chip tone={connected ? 'ok' : 'warn'} label="Live feed" value={connected ? 'connected' : 'reconnecting'} />
        <Chip
          tone={exchangeTone}
          label="Delta exchange"
          value={systemStatus == null || systemStatus === 'unknown' ? 'status unknown' : systemStatus}
        />
        <Chip
          tone={analystError ? 'bad' : analystAgeMin == null ? 'idle' : analystAgeMin > ANALYST_STALE_MINUTES ? 'warn' : 'ok'}
          label="Analyst"
          value={analystError ? 'unreachable' : analystUpdated ? `ran ${ago(analystUpdated)}` : 'not run yet'}
        />
      </div>
      {killSwitch && killReason && (
        <div role="alert" className="rounded-lg border border-red-900 bg-red-950/40 px-3 py-2 text-sm text-red-300">
          {killReason}
        </div>
      )}
    </section>
  )
}

// ------------------------------------------------------------------ positions

/** How far a level is from the current price, in percent. */
function distance(level: number | null | undefined, ltp: number | null): string | null {
  if (level == null || ltp == null || ltp === 0) return null
  return fmtPct((level / ltp - 1) * 100, 1)
}

function Level({ label, value, ltp, tone }: { label: string; value: number | null | undefined; ltp: number | null; tone: string }) {
  const d = distance(value, ltp)
  return (
    <div className="min-w-0 rounded-lg border border-neutral-800 bg-neutral-950/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`truncate font-mono text-[13px] tabular-nums ${value != null ? tone : 'text-neutral-500'}`}>{fmtPrice(value)}</div>
      <div className="font-mono text-[10px] text-neutral-400">{d ?? ' '}</div>
    </div>
  )
}

function PositionCard({ p, ltp }: { p: Position; ltp: number | null }) {
  const long = p.size > 0
  const onMargin = p.margin_usd > 0 ? (p.unrealized_pnl / p.margin_usd) * 100 : null
  return (
    <article className="rounded-lg border border-neutral-800 bg-neutral-950/40 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-semibold text-neutral-100">{coin(p.symbol)}</span>
            <span className={`text-sm font-semibold ${long ? 'text-emerald-400' : 'text-red-400'}`}>{long ? 'LONG' : 'SHORT'}</span>
          </div>
          <div className="text-xs text-neutral-400">
            {Math.abs(p.size)} lots{p.leverage != null ? ` · ${p.leverage}x` : ''} · margin {fmtUsd(p.margin_usd)}
          </div>
        </div>
        <div className="text-right">
          <div className={`font-mono text-xl font-semibold tabular-nums ${pnlClass(p.unrealized_pnl)}`}>
            {fmtUsd(p.unrealized_pnl, { signed: true })}
          </div>
          {onMargin != null && <div className={`font-mono text-[11px] ${pnlClass(onMargin)}`}>{fmtPct(onMargin, 1)} on margin</div>}
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Field label="Entry">
          <span className="text-neutral-200">{fmtPrice(p.entry_price)}</span>
        </Field>
        <Field label="Now" align="right">
          <span className="text-neutral-100">{fmtPrice(ltp)}</span>
        </Field>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2">
        <Level label="Stop loss" value={p.stop_loss_price ?? p.loss_exit_price} ltp={ltp} tone="text-red-300" />
        <Level label="Take profit" value={p.take_profit_price ?? p.profit_exit_price} ltp={ltp} tone="text-emerald-300" />
        <Level label="Liquidation" value={p.liquidation_price} ltp={ltp} tone="text-amber-300" />
      </div>
      <div className="mt-2 text-[11px] text-neutral-400">
        {p.strategy ?? 'no strategy'} · opened {formatISTShort(p.entry_time)} · fees {fmtUsd(p.fees_paid, { digits: 4 })}
      </div>
    </article>
  )
}

const TH = 'py-2 pr-3 text-left text-xs font-medium text-neutral-400'
const TD = 'py-2 pr-3 font-mono tabular-nums'

function PositionsTable({ positions, prices }: { positions: Position[]; prices: Record<string, number | null> }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className={TH}>Symbol</th>
            <th className={TH}>Side · lots</th>
            <th className={TH}>Lev.</th>
            <th className={TH}>Margin</th>
            <th className={TH}>Entry</th>
            <th className={TH}>Now</th>
            <th className={TH} title="Price at which the bot closes the trade at a loss">Stop loss</th>
            <th className={TH} title="Price at which the bot closes the trade in profit">Take profit</th>
            <th className={TH} title="Delta's liquidation price for this position">Liquidation</th>
            <th className={`${TH} text-right`}>Open P&amp;L</th>
            <th className={TH}>Strategy · opened</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => {
            const ltp = prices[p.symbol] ?? null
            const sl = p.stop_loss_price ?? p.loss_exit_price
            const tp = p.take_profit_price ?? p.profit_exit_price
            return (
              <tr key={`${p.symbol}-${p.strategy}`} className="border-b border-neutral-800/60">
                <td className="py-2 pr-3 font-medium text-neutral-100">{p.symbol}</td>
                <td className={`py-2 pr-3 font-medium ${p.size > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.size > 0 ? 'LONG' : 'SHORT'} {Math.abs(p.size)}
                </td>
                <td className={`${TD} text-neutral-300`}>{p.leverage != null ? `${p.leverage}x` : '--'}</td>
                <td className={`${TD} text-neutral-300`}>{fmtUsd(p.margin_usd)}</td>
                <td className={`${TD} text-neutral-300`}>{fmtPrice(p.entry_price)}</td>
                <td className={`${TD} text-neutral-100`}>{fmtPrice(ltp)}</td>
                <td className={`${TD} text-red-300`}>
                  {fmtPrice(sl)} <Dist v={distance(sl, ltp)} />
                </td>
                <td className={`${TD} text-emerald-300`}>
                  {fmtPrice(tp)} <Dist v={distance(tp, ltp)} />
                </td>
                <td className={`${TD} text-amber-300`}>
                  {fmtPrice(p.liquidation_price)} <Dist v={distance(p.liquidation_price, ltp)} />
                </td>
                <td className={`${TD} text-right font-medium ${pnlClass(p.unrealized_pnl)}`}>
                  {fmtUsd(p.unrealized_pnl, { signed: true })}
                </td>
                <td className="py-2 pr-3 text-xs text-neutral-400">
                  {p.strategy ?? '--'}
                  <div>{formatISTShort(p.entry_time)}</div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Dist({ v }: { v: string | null }) {
  return v ? <span className="block text-[10px] text-neutral-400">{v}</span> : null
}

// ------------------------------------------------------------------ closed trades

function ExitBadge({ reason }: { reason: string }) {
  const r = exitReason(reason)
  return <span className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] ${r.cls}`}>{r.label}</span>
}

function ClosedTradeItem({ t }: { t: ClosedTradeRow }) {
  const pnl = netPnl(t)
  return (
    <li className="flex items-center justify-between gap-3 py-2.5">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-neutral-100">{coin(t.symbol)}</span>
          <span className={`text-xs font-semibold ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side.toUpperCase()}</span>
          <ExitBadge reason={t.exit_reason} />
        </div>
        <div className="mt-0.5 text-[11px] text-neutral-400">
          closed {formatISTShort(t.exit_time)} · {t.size} lots
        </div>
      </div>
      <div className="text-right">
        <div className={`font-mono text-sm font-semibold tabular-nums ${pnlClass(pnl)}`}>{fmtUsd(pnl, { signed: true })}</div>
        <div className="font-mono text-[10px] text-neutral-400">fees {fmtUsd(t.fees, { digits: 4 })}</div>
      </div>
    </li>
  )
}

function ClosedTradesTable({ trades }: { trades: ClosedTradeRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[760px] text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className={TH}>Symbol</th>
            <th className={TH}>Side</th>
            <th className={TH}>Strategy</th>
            <th className={TH}>Lots</th>
            <th className={TH}>Opened</th>
            <th className={TH}>Closed</th>
            <th className={TH}>Exit</th>
            <th className={TH}>Fees</th>
            <th className={`${TH} text-right`}>P&amp;L (net)</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.id} className="border-b border-neutral-800/60">
              <td className="py-2 pr-3 font-medium text-neutral-100">{t.symbol}</td>
              <td className={`py-2 pr-3 font-medium ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side.toUpperCase()}</td>
              <td className="py-2 pr-3 text-neutral-300">{t.strategy ?? '--'}</td>
              <td className={`${TD} text-neutral-300`}>{t.size}</td>
              <td className="py-2 pr-3 text-xs text-neutral-400">{formatISTShort(t.entry_time)}</td>
              <td className="py-2 pr-3 text-xs text-neutral-400">{formatISTShort(t.exit_time)}</td>
              <td className="py-2 pr-3">
                <ExitBadge reason={t.exit_reason} />
              </td>
              <td className={`${TD} text-neutral-400`}>{fmtUsd(t.fees, { digits: 4 })}</td>
              <td className={`${TD} text-right font-medium ${pnlClass(netPnl(t))}`}>{fmtUsd(netPnl(t), { signed: true })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
