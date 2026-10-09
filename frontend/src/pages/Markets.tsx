import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { CandleChart } from '../components/CandleChart'
import { StatCard } from '../components/StatCard'
import { Empty, ErrorState, Loading, Panel } from '../components/ui'
import { useIsPhone } from '../lib/hooks'
import { fmtPrice } from '../lib/format'

const RESOLUTIONS = ['1m', '5m', '15m', '1h', '4h', '1d']

function usd(v: number | null | undefined): string {
  return v ? `$${fmtPrice(v)}` : '--'
}

export default function Markets() {
  const isPhone = useIsPhone()
  const symbolsQuery = useQuery({ queryKey: ['symbols'], queryFn: api.symbols })
  const [symbol, setSymbol] = useState<string | null>(null)
  const [resolution, setResolution] = useState('15m')

  const activeSymbol = symbol ?? symbolsQuery.data?.[0]?.symbol ?? null

  const tickerQuery = useQuery({
    queryKey: ['ticker', activeSymbol],
    queryFn: () => api.ticker(activeSymbol!),
    enabled: !!activeSymbol,
    refetchInterval: 3000,
  })
  const orderbookQuery = useQuery({
    queryKey: ['orderbook', activeSymbol],
    queryFn: () => api.orderbook(activeSymbol!),
    enabled: !!activeSymbol,
    refetchInterval: 2000,
  })
  const candlesQuery = useQuery({
    queryKey: ['candles', activeSymbol, resolution],
    queryFn: () => api.candles(activeSymbol!, resolution, 300),
    enabled: !!activeSymbol,
    refetchInterval: 15000,
  })

  const t = tickerQuery.data
  const funding = t?.funding_rate?.funding_rate
  const chartHeight = isPhone ? 280 : 360

  return (
    <div className="mx-auto max-w-7xl space-y-4 sm:space-y-6">
      {symbolsQuery.isError && (
        <ErrorState what="the symbol list" error={symbolsQuery.error} onRetry={() => symbolsQuery.refetch()} />
      )}
      {symbolsQuery.isLoading && <Loading className="h-9" label="Loading symbols" />}

      <div className="space-y-2 sm:flex sm:flex-wrap sm:items-center sm:gap-2 sm:space-y-0">
        {/* Symbols scroll sideways inside their own row on a phone. */}
        <div role="group" aria-label="Symbol" className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1 sm:mx-0 sm:flex-wrap sm:px-0 sm:pb-0">
          {(symbolsQuery.data ?? []).map((p) => (
            <button
              key={p.symbol}
              type="button"
              aria-pressed={activeSymbol === p.symbol}
              onClick={() => setSymbol(p.symbol)}
              className={`shrink-0 rounded-md border px-3 py-1.5 text-sm font-medium ${
                activeSymbol === p.symbol
                  ? 'border-emerald-500 bg-emerald-600 text-white'
                  : 'border-neutral-700 text-neutral-300 hover:text-neutral-100'
              }`}
            >
              {p.symbol}
            </button>
          ))}
        </div>
        <div role="group" aria-label="Candle size" className="flex gap-1 sm:ml-auto">
          {RESOLUTIONS.map((r) => (
            <button
              key={r}
              type="button"
              aria-pressed={resolution === r}
              onClick={() => setResolution(r)}
              className={`rounded px-2.5 py-1.5 text-xs font-medium ${
                resolution === r ? 'bg-neutral-700 text-neutral-50' : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {activeSymbol && (
        <>
          {tickerQuery.isError && <ErrorState what={`the ${activeSymbol} ticker`} error={tickerQuery.error} />}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4 md:gap-4">
            <StatCard label="Last price" value={usd(t?.last_price)} />
            <StatCard label="Mark price" value={usd(t?.mark_price)} />
            <StatCard label="Spot price" value={usd(t?.spot_price)} />
            <StatCard
              label="Funding rate"
              value={funding !== undefined && funding !== null ? `${(funding * 100).toFixed(4)}%` : '--'}
              tone={funding ? (funding > 0 ? 'positive' : 'negative') : 'default'}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Panel title={`${activeSymbol} · ${resolution}`} className="lg:col-span-2" bodyClassName="p-2 sm:p-4">
              {candlesQuery.isError ? (
                <ErrorState what="candles" error={candlesQuery.error} onRetry={() => candlesQuery.refetch()} />
              ) : candlesQuery.isLoading ? (
                <Loading className={isPhone ? 'h-[280px]' : 'h-[360px]'} label="Loading candles" />
              ) : candlesQuery.data && candlesQuery.data.length > 0 ? (
                <CandleChart candles={candlesQuery.data} height={chartHeight} />
              ) : (
                <Empty>No candles for this symbol yet</Empty>
              )}
            </Panel>

            <Panel title="Order book">
              {orderbookQuery.isError ? (
                <ErrorState what="the order book" error={orderbookQuery.error} />
              ) : orderbookQuery.isLoading ? (
                <Loading className="h-[200px]" label="Loading order book" />
              ) : orderbookQuery.data ? (
                <div className="grid grid-cols-2 gap-3 font-mono text-xs tabular-nums">
                  <div className="min-w-0">
                    <div className="mb-1 font-sans text-neutral-400">Bids</div>
                    {orderbookQuery.data.bids.slice(0, 10).map(([price, size]) => (
                      <div key={price} className="flex justify-between gap-2 py-0.5 text-emerald-400">
                        <span className="truncate">{fmtPrice(price)}</span>
                        <span className="text-neutral-400">{size.toFixed(0)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="min-w-0">
                    <div className="mb-1 font-sans text-neutral-400">Asks</div>
                    {orderbookQuery.data.asks.slice(0, 10).map(([price, size]) => (
                      <div key={price} className="flex justify-between gap-2 py-0.5 text-red-400">
                        <span className="truncate">{fmtPrice(price)}</span>
                        <span className="text-neutral-400">{size.toFixed(0)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="col-span-2 mt-1 border-t border-neutral-800 pt-2 font-sans text-neutral-300">
                    Imbalance:{' '}
                    {orderbookQuery.data.imbalance !== null ? (
                      <span className={`font-mono ${orderbookQuery.data.imbalance >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {(orderbookQuery.data.imbalance * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-neutral-400">--</span>
                    )}
                  </div>
                </div>
              ) : (
                <Empty>No book data yet</Empty>
              )}
            </Panel>
          </div>
        </>
      )}
    </div>
  )
}
