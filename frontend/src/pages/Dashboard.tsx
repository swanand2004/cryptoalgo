import { useQuery } from '@tanstack/react-query'
import { api, netPnl } from '../lib/api'
import { useDashboardSocket } from '../lib/ws'
import { EquityChart } from '../components/EquityChart'
import { StatCard } from '../components/StatCard'
import { formatISTShort, isTodayIST } from '../lib/time'

const RECENT_CLOSED_TRADES_LIMIT = 10

export default function Dashboard() {
  const { snapshot } = useDashboardSocket()
  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })
  const mode = modeQuery.data?.mode ?? 'paper'

  const equityCurveQuery = useQuery({
    queryKey: ['equity_curve', mode],
    queryFn: () => api.equityCurve(mode),
    refetchInterval: 15000,
  })
  // REST is only the fallback for first paint / if the socket drops -- once
  // connected, the WS snapshot (pushed every second) drives the table so
  // unrealized PnL tracks live price ticks instead of a slow 10s poll.
  const positionsQuery = useQuery({
    queryKey: ['positions'],
    queryFn: api.positions,
    refetchInterval: 10000,
  })
  const positions = snapshot?.positions ?? positionsQuery.data ?? []

  const closedTradesQuery = useQuery({
    queryKey: ['closed_trades', mode],
    queryFn: () => api.closedTrades(mode),
    refetchInterval: 10000,
  })
  const recentClosedTrades = (closedTradesQuery.data ?? []).slice(0, RECENT_CLOSED_TRADES_LIMIT)

  const equity = snapshot?.equity ?? equityCurveQuery.data?.at(-1)?.equity ?? 0

  const unrealizedPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0)
  // Realized PnL is scoped to trades that actually CLOSED today (IST) --
  // not all-time history -- and net of fees, so it reflects money actually
  // kept rather than just gross price movement. Open positions' own
  // realized_pnl (from a partial close) has no per-close timestamp to check
  // against "today", so it's left out here rather than risk mixing in
  // stale, undated gains.
  const realizedPnl = (closedTradesQuery.data ?? [])
    .filter((t) => isTodayIST(t.exit_time))
    .reduce((sum, t) => sum + netPnl(t), 0)
  // Same "today" scoping as realizedPnl, for the same reason -- open
  // positions' fees_paid has no per-fee timestamp to check against "today",
  // so it's left out here too rather than mixing in undated history.
  const totalFees = (closedTradesQuery.data ?? [])
    .filter((t) => isTodayIST(t.exit_time))
    .reduce((sum, t) => sum + t.fees, 0)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatCard
          label="Equity"
          value={`$${equity.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
        />
        <StatCard
          label="Unrealized PnL"
          value={`${unrealizedPnl >= 0 ? '+' : ''}$${unrealizedPnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
          tone={unrealizedPnl >= 0 ? 'positive' : 'negative'}
        />
        <StatCard
          label="Realized PnL (Today)"
          value={`${realizedPnl >= 0 ? '+' : ''}$${realizedPnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
          tone={realizedPnl >= 0 ? 'positive' : 'negative'}
        />
        <StatCard
          label="Total Fees (Today)"
          value={`$${totalFees.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
        />
        <StatCard
          label="Open Positions"
          value={String(positions.length)}
          sub={snapshot?.system_status ? `system: ${snapshot.system_status}` : undefined}
        />
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <div className="text-sm font-medium text-neutral-300 mb-3">Positions</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-neutral-500 border-b border-neutral-800">
              <th className="py-2 font-medium">Symbol</th>
              <th className="py-2 font-medium">Side</th>
              <th className="py-2 font-medium">Size (Lots)</th>
              <th className="py-2 font-medium">Strategy</th>
              <th className="py-2 font-medium">Leverage</th>
              <th className="py-2 font-medium">Margin (USD)</th>
              <th className="py-2 font-medium">Entry Time</th>
              <th className="py-2 font-medium">Entry</th>
              <th className="py-2 font-medium">LTP</th>
              <th className="py-2 font-medium">Stop Loss</th>
              <th className="py-2 font-medium">Take Profit</th>
              <th className="py-2 font-medium">Unrealized PnL</th>
              <th className="py-2 font-medium">Fees</th>
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 && (
              <tr>
                <td colSpan={13} className="py-6 text-center text-neutral-600">
                  No open positions
                </td>
              </tr>
            )}
            {positions.map((p) => (
              <tr key={`${p.symbol}-${p.strategy}`} className="border-b border-neutral-900">
                <td className="py-2 font-medium text-neutral-200">{p.symbol}</td>
                <td className={`py-2 font-medium ${p.size > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.size > 0 ? 'LONG' : 'SHORT'}
                </td>
                <td className={`py-2 font-mono ${p.size > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.size > 0 ? '+' : ''}
                  {p.size}
                </td>
                <td className="py-2 text-neutral-400">{p.strategy ?? '--'}</td>
                <td className="py-2 font-mono text-neutral-400">
                  {p.leverage !== null ? `${p.leverage}x` : '--'}
                </td>
                <td className="py-2 font-mono text-neutral-400">
                  ${p.margin_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </td>
                <td className="py-2 text-neutral-500 font-mono text-xs">
                  {formatISTShort(p.entry_time)}
                </td>
                <td className="py-2 font-mono text-neutral-400">{p.entry_price.toFixed(2)}</td>
                <td className="py-2 font-mono text-neutral-200">
                  {snapshot?.prices[p.symbol] != null ? snapshot.prices[p.symbol]!.toFixed(2) : '--'}
                </td>
                <td className="py-2 font-mono text-red-400">
                  {p.stop_loss_price !== null ? p.stop_loss_price.toFixed(2) : '--'}
                </td>
                <td className="py-2 font-mono text-emerald-400">
                  {p.take_profit_price !== null ? p.take_profit_price.toFixed(2) : '--'}
                </td>
                <td className={`py-2 font-mono ${p.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.unrealized_pnl >= 0 ? '+' : ''}
                  {p.unrealized_pnl.toFixed(2)}
                </td>
                <td className="py-2 font-mono text-neutral-500">{p.fees_paid.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <div className="text-sm font-medium text-neutral-300 mb-3">Recent Closed Trades ({mode})</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-neutral-500 border-b border-neutral-800">
              <th className="py-2 font-medium">Symbol</th>
              <th className="py-2 font-medium">Side</th>
              <th className="py-2 font-medium">Strategy</th>
              <th className="py-2 font-medium">Size (Lots)</th>
              <th className="py-2 font-medium">Entry Time</th>
              <th className="py-2 font-medium">Exit Time</th>
              <th className="py-2 font-medium">Exit Reason</th>
              <th className="py-2 font-medium">Fees</th>
              <th className="py-2 font-medium">PnL (Net of Fees)</th>
            </tr>
          </thead>
          <tbody>
            {recentClosedTrades.length === 0 && (
              <tr>
                <td colSpan={9} className="py-6 text-center text-neutral-600">
                  No closed trades yet in {mode} mode
                </td>
              </tr>
            )}
            {recentClosedTrades.map((t) => (
              <tr key={t.id} className="border-b border-neutral-900">
                <td className="py-2 font-medium text-neutral-200">{t.symbol}</td>
                <td className={`py-2 font-medium ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {t.side.toUpperCase()}
                </td>
                <td className="py-2 text-neutral-400">{t.strategy ?? '--'}</td>
                <td className="py-2 font-mono">{t.size}</td>
                <td className="py-2 text-neutral-500 font-mono text-xs">{formatISTShort(t.entry_time)}</td>
                <td className="py-2 text-neutral-500 font-mono text-xs">{formatISTShort(t.exit_time)}</td>
                <td className="py-2">
                  <span
                    className={`px-2 py-0.5 rounded text-xs ${
                      t.exit_reason === 'stop_loss'
                        ? 'bg-red-950 text-red-400'
                        : t.exit_reason === 'take_profit'
                        ? 'bg-emerald-950 text-emerald-400'
                        : 'bg-neutral-800 text-neutral-400'
                    }`}
                  >
                    {t.exit_reason === 'stop_loss' ? 'STOP LOSS' : t.exit_reason === 'take_profit' ? 'TAKE PROFIT' : 'SIGNAL'}
                  </span>
                </td>
                <td className="py-2 font-mono text-neutral-500">{t.fees.toFixed(4)}</td>
                <td className={`py-2 font-mono font-medium ${netPnl(t) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {netPnl(t) >= 0 ? '+' : ''}
                  {netPnl(t).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <div className="text-sm font-medium text-neutral-300 mb-3">Equity Curve ({mode})</div>
        {equityCurveQuery.data && equityCurveQuery.data.length > 1 ? (
          <EquityChart points={equityCurveQuery.data} />
        ) : (
          <div className="h-[260px] flex items-center justify-center text-neutral-600 text-sm">
            Collecting equity history -- snapshots are recorded every minute.
          </div>
        )}
      </div>
    </div>
  )
}
