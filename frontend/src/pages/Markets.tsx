import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { CandleChart } from '../components/CandleChart'
import { StatCard } from '../components/StatCard'

const RESOLUTIONS = ['1m', '5m', '15m', '1h', '4h', '1d']

export default function Markets() {
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

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        {(symbolsQuery.data ?? []).map((p) => (
          <button
            key={p.symbol}
            onClick={() => setSymbol(p.symbol)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium border ${
              activeSymbol === p.symbol
                ? 'bg-emerald-600 border-emerald-500 text-white'
                : 'border-neutral-700 text-neutral-400 hover:text-neutral-200'
            }`}
          >
            {p.symbol}
          </button>
        ))}
        <div className="ml-auto flex gap-1">
          {RESOLUTIONS.map((r) => (
            <button
              key={r}
              onClick={() => setResolution(r)}
              className={`px-2.5 py-1 rounded text-xs font-medium ${
                resolution === r ? 'bg-neutral-800 text-neutral-100' : 'text-neutral-500 hover:text-neutral-300'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {activeSymbol && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Last Price" value={t?.last_price ? `$${t.last_price.toLocaleString()}` : '--'} />
            <StatCard label="Mark Price" value={t?.mark_price ? `$${t.mark_price.toLocaleString()}` : '--'} />
            <StatCard label="Spot Price" value={t?.spot_price ? `$${t.spot_price.toLocaleString()}` : '--'} />
            <StatCard
              label="Funding Rate"
              value={funding !== undefined && funding !== null ? `${(funding * 100).toFixed(4)}%` : '--'}
              tone={funding ? (funding > 0 ? 'positive' : 'negative') : 'default'}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 rounded-lg border border-neutral-800 bg-neutral-900 p-4">
              <div className="text-sm font-medium text-neutral-300 mb-3">
                {activeSymbol} &middot; {resolution}
              </div>
              {candlesQuery.data && candlesQuery.data.length > 0 ? (
                <CandleChart candles={candlesQuery.data} />
              ) : (
                <div className="h-[360px] flex items-center justify-center text-neutral-600 text-sm">
                  Loading candles...
                </div>
              )}
            </div>

            <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
              <div className="text-sm font-medium text-neutral-300 mb-3">Order Book</div>
              {orderbookQuery.data ? (
                <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                  <div>
                    <div className="text-neutral-500 mb-1">Bids</div>
                    {orderbookQuery.data.bids.slice(0, 10).map(([price, size]) => (
                      <div key={price} className="flex justify-between text-emerald-400 py-0.5">
                        <span>{price.toFixed(2)}</span>
                        <span className="text-neutral-500">{size.toFixed(0)}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div className="text-neutral-500 mb-1">Asks</div>
                    {orderbookQuery.data.asks.slice(0, 10).map(([price, size]) => (
                      <div key={price} className="flex justify-between text-red-400 py-0.5">
                        <span>{price.toFixed(2)}</span>
                        <span className="text-neutral-500">{size.toFixed(0)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="col-span-2 pt-2 mt-1 border-t border-neutral-800 text-neutral-400">
                    Imbalance:{' '}
                    {orderbookQuery.data.imbalance !== null ? (
                      <span className={orderbookQuery.data.imbalance >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                        {(orderbookQuery.data.imbalance * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="text-neutral-600">--</span>
                    )}
                  </div>
                </div>
              ) : (
                <div className="h-[200px] flex items-center justify-center text-neutral-600 text-sm">
                  No book data yet
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
