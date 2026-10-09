import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, netPnl, type ClosedTradeRow, type OrderRow } from '../lib/api'
import { formatISTShort, istDateEndUtc, istDateStartUtc } from '../lib/time'
import { coin, exitReason, fmtPrice, fmtUsd, pnlClass } from '../lib/format'
import { Empty, ErrorState, Loading } from '../components/ui'
import { useIsPhone } from '../lib/hooks'

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

const TH = 'py-2 px-2 text-left text-xs font-medium text-neutral-400'

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="group" aria-label={label} className="flex overflow-hidden rounded-md border border-neutral-700 text-sm">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={`px-3 py-1.5 font-medium ${
            value === o.value ? 'bg-neutral-700 text-neutral-50' : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export default function Trades() {
  const isPhone = useIsPhone()
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
  const [view, setView] = useState<'trades' | 'orders'>('trades')
  // Plain YYYY-MM-DD strings from <input type="date">, interpreted as IST
  // calendar dates (matching how every timestamp elsewhere is displayed) --
  // converted to UTC instants only at the query boundary.
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  const tradesQuery = useQuery({
    queryKey: ['closed_trades', mode, dateFrom, dateTo],
    queryFn: () =>
      api.closedTrades(
        mode,
        undefined,
        dateFrom ? istDateStartUtc(dateFrom) : undefined,
        dateTo ? istDateEndUtc(dateTo) : undefined
      ),
    enabled: view === 'trades',
    refetchInterval: 5000,
  })
  const ordersQuery = useQuery({
    queryKey: ['orders', mode],
    queryFn: () => api.orders(mode),
    enabled: view === 'orders',
    refetchInterval: 5000,
  })

  const trades = tradesQuery.data ?? []
  const totalNet = trades.reduce((sum, t) => sum + netPnl(t), 0)
  const wins = trades.filter((t) => netPnl(t) > 0).length

  return (
    // Wider than the other pages -- the desktop trades table has many
    // columns. On a phone the rows become cards instead.
    <div className="mx-auto max-w-[1800px] space-y-4">
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <Segmented
          label="Account"
          value={mode}
          onChange={setModeOverride}
          options={[
            { value: 'paper', label: 'Paper' },
            { value: 'live', label: 'Live' },
          ]}
        />
        <Segmented
          label="View"
          value={view}
          onChange={setView}
          options={[
            { value: 'trades', label: 'Trades' },
            { value: 'orders', label: 'Orders' },
          ]}
        />

        {view === 'trades' && (
          <div className="flex w-full flex-wrap items-center gap-2 text-sm sm:ml-auto sm:w-auto">
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
              From
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                max={dateTo || undefined}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-200 [color-scheme:dark]"
              />
            </label>
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
              To
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                min={dateFrom || undefined}
                className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm text-neutral-200 [color-scheme:dark]"
              />
            </label>
            {(dateFrom || dateTo) && (
              <button
                type="button"
                onClick={() => {
                  setDateFrom('')
                  setDateTo('')
                }}
                className="px-1 py-1 text-xs text-neutral-300 underline hover:text-neutral-100"
              >
                Clear dates
              </button>
            )}
          </div>
        )}
      </div>

      {mode !== actualMode && (
        <div className="text-xs text-amber-300">
          Viewing {mode.toUpperCase()} history -- the account is currently {actualMode.toUpperCase()}.
        </div>
      )}

      {view === 'trades' && tradesQuery.isSuccess && trades.length > 0 && (
        <div className="flex flex-wrap gap-x-5 gap-y-1 rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-sm text-neutral-300">
          <span>
            {/* The API returns at most the latest 100 closed trades. */}
            {trades.length >= 100 ? 'latest ' : ''}
            <span className="font-mono text-neutral-100">{trades.length}</span> trades
          </span>
          <span>
            win <span className="font-mono text-neutral-100">{Math.round((wins / trades.length) * 100)}%</span>
          </span>
          <span>
            net <span className={`font-mono font-semibold ${pnlClass(totalNet)}`}>{fmtUsd(totalNet, { signed: true })}</span>
          </span>
        </div>
      )}

      <div className="min-w-0 rounded-xl border border-neutral-800 bg-neutral-900">
        {view === 'trades' &&
          (tradesQuery.isError ? (
            <div className="p-3">
              <ErrorState what="closed trades" error={tradesQuery.error} onRetry={() => tradesQuery.refetch()} />
            </div>
          ) : tradesQuery.isLoading ? (
            <div className="p-3">
              <Loading className="h-40" label="Loading trades" />
            </div>
          ) : trades.length === 0 ? (
            <Empty>
              No closed trades in {mode} mode{dateFrom || dateTo ? ' for the selected dates' : ' yet'}
            </Empty>
          ) : isPhone ? (
            <ul className="divide-y divide-neutral-800 px-3">
              {trades.map((t) => (
                <TradeCard key={t.id} t={t} />
              ))}
            </ul>
          ) : (
            <TradesTable trades={trades} />
          ))}

        {view === 'orders' &&
          (ordersQuery.isError ? (
            <div className="p-3">
              <ErrorState what="orders" error={ordersQuery.error} onRetry={() => ordersQuery.refetch()} />
            </div>
          ) : ordersQuery.isLoading ? (
            <div className="p-3">
              <Loading className="h-40" label="Loading orders" />
            </div>
          ) : (ordersQuery.data ?? []).length === 0 ? (
            <Empty>No orders yet in {mode} mode</Empty>
          ) : isPhone ? (
            <ul className="divide-y divide-neutral-800 px-3">
              {(ordersQuery.data ?? []).map((o) => (
                <OrderCard key={o.id} o={o} />
              ))}
            </ul>
          ) : (
            <OrdersTable orders={ordersQuery.data ?? []} />
          ))}
      </div>
    </div>
  )
}

function ExitBadge({ reason }: { reason: string }) {
  const r = exitReason(reason)
  return <span className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] ${r.cls}`}>{r.label}</span>
}

function TradeCard({ t }: { t: ClosedTradeRow }) {
  const pnl = netPnl(t)
  return (
    <li className="py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-medium text-neutral-100">{coin(t.symbol)}</span>
            <span className={`text-xs font-semibold ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side.toUpperCase()}</span>
            <ExitBadge reason={t.exit_reason} />
          </div>
          <div className="mt-0.5 text-[11px] text-neutral-400">
            {t.size} lots{t.leverage != null ? ` · ${t.leverage}x` : ''} · {formatDuration(t.entry_time, t.exit_time)}
            {t.strategy ? ` · ${t.strategy}` : ''}
          </div>
        </div>
        <div className="text-right">
          <div className={`font-mono text-sm font-semibold tabular-nums ${pnlClass(pnl)}`}>{fmtUsd(pnl, { signed: true })}</div>
          <div className="font-mono text-[10px] text-neutral-400">fees {fmtUsd(t.fees, { digits: 4 })}</div>
        </div>
      </div>
      <div className="mt-1.5 grid grid-cols-2 gap-2 text-[12px]">
        <div className="min-w-0">
          <div className="text-[10px] uppercase tracking-wide text-neutral-400">In</div>
          <div className="truncate font-mono tabular-nums text-neutral-200">{fmtPrice(t.entry_price)}</div>
          <div className="text-[11px] text-neutral-400">{formatISTShort(t.entry_time)}</div>
        </div>
        <div className="min-w-0 text-right">
          <div className="text-[10px] uppercase tracking-wide text-neutral-400">Out</div>
          <div className="truncate font-mono tabular-nums text-neutral-200">{fmtPrice(t.exit_price)}</div>
          <div className="text-[11px] text-neutral-400">{formatISTShort(t.exit_time)}</div>
        </div>
      </div>
    </li>
  )
}

function TradesTable({ trades }: { trades: ClosedTradeRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1100px] text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className={TH}>Symbol</th>
            <th className={TH}>Side</th>
            <th className={TH}>Strategy</th>
            <th className={TH}>Lev.</th>
            <th className={TH}>Lots</th>
            <th className={TH}>Margin</th>
            <th className={TH}>Opened</th>
            <th className={TH}>Entry</th>
            <th className={TH}>Closed</th>
            <th className={TH}>Exit</th>
            <th className={TH}>Held</th>
            <th className={TH}>Reason</th>
            <th className={TH}>Fees</th>
            <th className={`${TH} text-right`}>P&amp;L (net)</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.id} className="border-b border-neutral-800/60">
              <td className="px-2 py-2 font-medium text-neutral-100">{t.symbol}</td>
              <td className={`px-2 py-2 font-medium ${t.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side.toUpperCase()}</td>
              <td className="px-2 py-2 text-neutral-300">{t.strategy ?? '--'}</td>
              <td className="px-2 py-2 font-mono text-neutral-300">{t.leverage !== null ? `${t.leverage}x` : '--'}</td>
              <td className="px-2 py-2 font-mono text-neutral-200">{t.size}</td>
              <td className="px-2 py-2 font-mono text-neutral-300">{t.margin_usd !== null ? fmtUsd(t.margin_usd) : '--'}</td>
              <td className="px-2 py-2 text-xs text-neutral-400">{formatISTShort(t.entry_time)}</td>
              <td className="px-2 py-2 font-mono tabular-nums text-neutral-200">{fmtPrice(t.entry_price)}</td>
              <td className="px-2 py-2 text-xs text-neutral-400">{formatISTShort(t.exit_time)}</td>
              <td className="px-2 py-2 font-mono tabular-nums text-neutral-200">{fmtPrice(t.exit_price)}</td>
              <td className="px-2 py-2 font-mono text-xs text-neutral-300">{formatDuration(t.entry_time, t.exit_time)}</td>
              <td className="px-2 py-2">
                <ExitBadge reason={t.exit_reason} />
              </td>
              <td className="px-2 py-2 font-mono text-neutral-400">{fmtUsd(t.fees, { digits: 4 })}</td>
              <td className={`px-2 py-2 text-right font-mono font-medium tabular-nums ${pnlClass(netPnl(t))}`}>
                {fmtUsd(netPnl(t), { signed: true })}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const ORDER_STATE: Record<string, string> = {
  closed: 'bg-emerald-950 text-emerald-300',
  cancelled: 'bg-red-950 text-red-300',
  rejected: 'bg-red-950 text-red-300',
}

function OrderState({ state }: { state: string }) {
  return <span className={`rounded px-1.5 py-0.5 text-[11px] ${ORDER_STATE[state] ?? 'bg-neutral-800 text-neutral-300'}`}>{state}</span>
}

function OrderCard({ o }: { o: OrderRow }) {
  return (
    <li className="flex items-start justify-between gap-3 py-2.5">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-neutral-100">{coin(o.symbol)}</span>
          <span className={`text-xs font-semibold ${o.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>{o.side.toUpperCase()}</span>
          <span className="text-xs text-neutral-400">{o.order_type}</span>
        </div>
        <div className="mt-0.5 text-[11px] text-neutral-400">
          {formatISTShort(o.created_at)}
          {o.strategy ? ` · ${o.strategy}` : ''}
        </div>
      </div>
      <div className="text-right">
        <OrderState state={o.state} />
        <div className="mt-1 font-mono text-[11px] text-neutral-300">
          {o.size} lots{o.unfilled_size ? ` · ${o.unfilled_size} unfilled` : ''}
        </div>
      </div>
    </li>
  )
}

function OrdersTable({ orders }: { orders: OrderRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-neutral-800">
            <th className={TH}>Time</th>
            <th className={TH}>Symbol</th>
            <th className={TH}>Side</th>
            <th className={TH}>Type</th>
            <th className={TH}>Size</th>
            <th className={TH}>Unfilled</th>
            <th className={TH}>State</th>
            <th className={TH}>Strategy</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.id} className="border-b border-neutral-800/60">
              <td className="px-2 py-2 text-xs text-neutral-400">{formatISTShort(o.created_at)}</td>
              <td className="px-2 py-2 font-medium text-neutral-100">{o.symbol}</td>
              <td className={`px-2 py-2 font-medium ${o.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>{o.side.toUpperCase()}</td>
              <td className="px-2 py-2 text-neutral-300">{o.order_type}</td>
              <td className="px-2 py-2 font-mono text-neutral-200">{o.size}</td>
              <td className="px-2 py-2 font-mono text-neutral-400">{o.unfilled_size}</td>
              <td className="px-2 py-2">
                <OrderState state={o.state} />
              </td>
              <td className="px-2 py-2 text-neutral-400">{o.strategy ?? '--'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
