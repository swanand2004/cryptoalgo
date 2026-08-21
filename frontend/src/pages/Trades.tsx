import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, netPnl } from '../lib/api'
import { formatISTShort } from '../lib/time'

function formatDuration(entryIso: string, exitIso: string): string {
  const ms = new Date(exitIso).getTime() - new Date(entryIso).getTime()
  const minutes = Math.round(ms / 60000)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60
  if (hours < 24) return `${hours}h ${remMinutes}m`
  const days = Math.floor(hours / 24)
  return `${days}d ${hours % 24}h`
}

export default function Trades() {
  // Defaults to whichever mode the account is ACTUALLY in right now (not a
  // hardcoded "paper") -- otherwise a user trading live could land here,
  // see "no trades yet in paper mode", and wrongly conclude nothing has
  // traded. Once the user manually picks a tab, that choice sticks (so they
  // can still review paper history while live, or vice versa) instead of
  // snapping back on every 5s mode poll.
  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })
  const [modeOverride, setModeOverride] = useState<'paper' | 'live' | null>(null)
  const actualMode = modeQuery.data?.mode ?? 'paper'
  const mode = modeOverride ?? (actualMode as 'paper' | 'live')
  const setMode = setModeOverride
  const [view, setView] = useState<'trades' | 'orders'>('trades')

  const tradesQuery = useQuery({
    queryKey: ['closed_trades', mode],
    queryFn: () => api.closedTrades(mode),
    enabled: view === 'trades',
    refetchInterval: 5000,
  })
  const ordersQuery = useQuery({
    queryKey: ['orders', mode],
    queryFn: () => api.orders(mode),
    enabled: view === 'orders',
    refetchInterval: 5000,
  })

  return (
    // Wider than the other pages' max-w-7xl -- the trades table has 14
    // columns and needs real room to fit without horizontal scrolling.
    // Layout.tsx applies no width constraint of its own (each page sets
    // its own), so this is just a normal, correctly-centering max-w/mx-auto
    // -- no viewport-relative positioning tricks needed.
    <div className="max-w-[1800px] mx-auto space-y-4">
      <div className="flex items-center gap-4">
        <div className="flex rounded-md border border-neutral-700 overflow-hidden text-sm">
          {(['paper', 'live'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 font-medium ${
                mode === m ? 'bg-neutral-800 text-neutral-100' : 'text-neutral-500 hover:text-neutral-300'
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="flex rounded-md border border-neutral-700 overflow-hidden text-sm">
          {(['trades', 'orders'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 font-medium ${
                view === v ? 'bg-neutral-800 text-neutral-100' : 'text-neutral-500 hover:text-neutral-300'
              }`}
            >
              {v === 'trades' ? 'Trades' : v}
            </button>
          ))}
        </div>
        {mode !== actualMode && (
          <div className="text-xs text-amber-400">
            Viewing {mode.toUpperCase()} history -- account is currently {actualMode.toUpperCase()}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900 overflow-x-auto">
        {view === 'trades' && (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-neutral-500 border-b border-neutral-800">
                <th className="py-2 px-2 font-medium">Symbol</th>
                <th className="py-2 px-2 font-medium">Side</th>
                <th className="py-2 px-2 font-medium">Strategy</th>
                <th className="py-2 px-2 font-medium">Leverage</th>
                <th className="py-2 px-2 font-medium">Size (Lots)</th>
                <th className="py-2 px-2 font-medium">Margin (USD)</th>
                <th className="py-2 px-2 font-medium">Entry Time</th>
                <th className="py-2 px-2 font-medium">Entry Price</th>
                <th className="py-2 px-2 font-medium">Exit Time</th>
                <th className="py-2 px-2 font-medium">Exit Price</th>
                <th className="py-2 px-2 font-medium">Duration</th>
                <th className="py-2 px-2 font-medium">Exit Reason</th>
                <th className="py-2 px-2 font-medium">Fees</th>
                <th className="py-2 px-2 font-medium">PnL (Net of Fees)</th>
              </tr>
            </thead>
            <tbody>
              {(tradesQuery.data ?? []).length === 0 && (
                <tr>
                  <td colSpan={14} className="py-6 text-center text-neutral-600">
                    No closed trades yet in {mode} mode
                  </td>
                </tr>
              )}
              {(tradesQuery.data ?? []).map((t) => (
                <tr key={t.id} className="border-b border-neutral-900">
                  <td className="py-2 px-2 font-medium text-neutral-200">{t.symbol}</td>
                  <td className={`py-2 px-2 font-medium ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {t.side.toUpperCase()}
                  </td>
                  <td className="py-2 px-2 text-neutral-400">{t.strategy ?? '--'}</td>
                  <td className="py-2 px-2 font-mono text-neutral-400">
                    {t.leverage !== null ? `${t.leverage}x` : '--'}
                  </td>
                  <td className="py-2 px-2 font-mono">{t.size}</td>
                  <td className="py-2 px-2 font-mono text-neutral-400">
                    {t.margin_usd !== null ? `$${t.margin_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '--'}
                  </td>
                  <td className="py-2 px-2 text-neutral-500 font-mono text-xs">
                    {formatISTShort(t.entry_time)}
                  </td>
                  <td className="py-2 px-2 font-mono">{t.entry_price.toFixed(2)}</td>
                  <td className="py-2 px-2 text-neutral-500 font-mono text-xs">
                    {formatISTShort(t.exit_time)}
                  </td>
                  <td className="py-2 px-2 font-mono">{t.exit_price.toFixed(2)}</td>
                  <td className="py-2 px-2 text-neutral-400 font-mono text-xs">
                    {formatDuration(t.entry_time, t.exit_time)}
                  </td>
                  <td className="py-2 px-2">
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
                  <td className="py-2 px-2 font-mono text-neutral-500">{t.fees.toFixed(4)}</td>
                  <td className={`py-2 px-2 font-mono font-medium ${netPnl(t) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {netPnl(t) >= 0 ? '+' : ''}
                    {netPnl(t).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {view === 'orders' && (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-neutral-500 border-b border-neutral-800">
                <th className="py-2 px-2 font-medium">Time</th>
                <th className="py-2 px-2 font-medium">Symbol</th>
                <th className="py-2 px-2 font-medium">Side</th>
                <th className="py-2 px-2 font-medium">Type</th>
                <th className="py-2 px-2 font-medium">Size</th>
                <th className="py-2 px-2 font-medium">Unfilled</th>
                <th className="py-2 px-2 font-medium">State</th>
                <th className="py-2 px-2 font-medium">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {(ordersQuery.data ?? []).length === 0 && (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-neutral-600">
                    No orders yet in {mode} mode
                  </td>
                </tr>
              )}
              {(ordersQuery.data ?? []).map((o) => (
                <tr key={o.id} className="border-b border-neutral-900">
                  <td className="py-2 px-2 text-neutral-500 font-mono text-xs">
                    {formatISTShort(o.created_at)}
                  </td>
                  <td className="py-2 px-2 font-medium text-neutral-200">{o.symbol}</td>
                  <td className={`py-2 px-2 font-medium ${o.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {o.side.toUpperCase()}
                  </td>
                  <td className="py-2 px-2 text-neutral-400">{o.order_type}</td>
                  <td className="py-2 px-2 font-mono">{o.size}</td>
                  <td className="py-2 px-2 font-mono text-neutral-500">{o.unfilled_size}</td>
                  <td className="py-2 px-2">
                    <span
                      className={`px-2 py-0.5 rounded text-xs ${
                        o.state === 'closed'
                          ? 'bg-emerald-950 text-emerald-400'
                          : o.state === 'cancelled' || o.state === 'rejected'
                          ? 'bg-red-950 text-red-400'
                          : 'bg-neutral-800 text-neutral-400'
                      }`}
                    >
                      {o.state}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-neutral-500">{o.strategy ?? '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
