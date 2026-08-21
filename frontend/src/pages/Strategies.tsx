import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, netPnl, type ClosedTradeRow } from '../lib/api'

// Only the 3 strategies actually wired into the live system (see
// backend/app/strategy/registry.py's DEFAULT_WEIGHTS). Everything else was
// backtested, cut on results, and fully removed from the running system --
// not just disabled -- so there's nothing left to describe here for them.
const DESCRIPTIONS: Record<string, string> = {
  trend_momentum: 'EMA cross + MACD confirmation, ATR-normalized. Rides trends.',
  graha: 'EXPERIMENTAL, deep Panchang mechanism. Real ephemeris math (JPL DE421 + Lahiri ayanamsa + Meeus lunar nodes) covering all 9 Navagraha: gated flat during Rahu Kaal, Vishti Karana, or a severe Nitya Yoga; tilted by lunar paksha, nakshatra-lord benefic/malefic, and planetary aspects to the Moon; discounted during Mercury/Mars/Saturn retrograde. No established causal link between astrology and markets -- kept enabled as an explicit, informed override, judged purely on backtest results.',
  tsmom: 'Time-series momentum: a barbell of a short + long lookback horizon, each voting on the sign of its own trailing return (vol-normalized). Grounded in Moskowitz/Ooi/Pedersen (2012) and AQR’s "Century of Evidence on Trend-Following."',
}

interface SymbolStats {
  symbol: string
  trades: number
  wins: number
  winRate: number
  pnl: number
}

interface StrategyStats {
  strategy: string
  trades: number
  wins: number
  winRate: number
  pnl: number
  bySymbol: SymbolStats[]
}

function groupPerformance(trades: ClosedTradeRow[]): StrategyStats[] {
  const byStrategy = new Map<string, Map<string, ClosedTradeRow[]>>()
  for (const t of trades) {
    const strategy = t.strategy ?? 'manual'
    if (!byStrategy.has(strategy)) byStrategy.set(strategy, new Map())
    const bySymbol = byStrategy.get(strategy)!
    if (!bySymbol.has(t.symbol)) bySymbol.set(t.symbol, [])
    bySymbol.get(t.symbol)!.push(t)
  }

  const summarize = (rows: ClosedTradeRow[]) => {
    // Win/loss and PnL are both net of fees -- a trade that made money on
    // price but lost money after commissions is a loss, not a win.
    const wins = rows.filter((t) => netPnl(t) > 0).length
    const pnl = rows.reduce((sum, t) => sum + netPnl(t), 0)
    return { trades: rows.length, wins, winRate: rows.length ? Math.round((wins / rows.length) * 1000) / 10 : 0, pnl }
  }

  return Array.from(byStrategy.entries())
    .map(([strategy, bySymbolMap]) => {
      const allRows = Array.from(bySymbolMap.values()).flat()
      const bySymbol = Array.from(bySymbolMap.entries())
        .map(([symbol, rows]) => ({ symbol, ...summarize(rows) }))
        .sort((a, b) => b.trades - a.trades)
      return { strategy, ...summarize(allRows), bySymbol }
    })
    .sort((a, b) => b.trades - a.trades)
}

// Fixed categorical order/hues (dark-surface steps, validated all-pairs
// CVD-safe) -- color always maps to the same symbol everywhere, never
// reassigned based on which symbols happen to have trades.
const SYMBOL_COLORS: Record<string, string> = {
  BTCUSD: '#3987e5', // blue
  ETHUSD: '#d95926', // orange
  SOLUSD: '#199e70', // aqua
}
const SYMBOL_ORDER = ['BTCUSD', 'ETHUSD', 'SOLUSD']

function PerformanceChart({ stats }: { stats: StrategyStats[] }) {
  const presentSymbols = SYMBOL_ORDER.filter((sym) => stats.some((s) => s.bySymbol.some((b) => b.symbol === sym)))
  if (presentSymbols.length === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs text-neutral-400">
        {presentSymbols.map((sym) => (
          <div key={sym} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: SYMBOL_COLORS[sym] }} />
            {sym}
          </div>
        ))}
      </div>

      <div className="space-y-3">
        {stats.map((s) => (
          <div key={s.strategy} className="flex items-center gap-3">
            <div className="w-32 flex-shrink-0 text-xs text-neutral-400 truncate">{s.strategy}</div>
            <div className="flex-1 space-y-1">
              {presentSymbols.map((sym) => {
                const b = s.bySymbol.find((x) => x.symbol === sym)
                const winRate = b?.winRate ?? 0
                return (
                  <div key={sym} className="group flex items-center gap-2">
                    <div className="flex-1 h-2.5 rounded-full bg-neutral-800 overflow-hidden">
                      {b && (
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.max(winRate, 2)}%`, backgroundColor: SYMBOL_COLORS[sym] }}
                          title={`${sym} / ${s.strategy}: ${winRate}% win rate over ${b.trades} trades, ${b.pnl >= 0 ? '+' : ''}$${b.pnl.toFixed(2)} PnL`}
                        />
                      )}
                    </div>
                    <div className="w-16 flex-shrink-0 font-mono text-xs text-neutral-500 text-right">
                      {b ? `${winRate}%` : '--'}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PerformanceSection() {
  // Same fix as Trades.tsx: default to the account's ACTUAL current mode
  // rather than a hardcoded "paper", so this doesn't silently show stale/
  // wrong-mode performance. A manual pick still sticks.
  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })
  const [modeOverride, setModeOverride] = useState<'paper' | 'live' | null>(null)
  const actualMode = modeQuery.data?.mode ?? 'paper'
  const mode = modeOverride ?? (actualMode as 'paper' | 'live')
  const tradesQuery = useQuery({
    queryKey: ['closed_trades_performance', mode],
    queryFn: () => api.closedTrades(mode),
    refetchInterval: 15000,
  })

  const stats = groupPerformance(tradesQuery.data ?? [])

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-neutral-300">Performance by Strategy &amp; Instrument</div>
          <div className="text-xs text-neutral-500 mt-0.5">
            Win rate and PnL from this account's actual closed trades -- real fills, not a hypothetical backtest.
          </div>
          {mode !== actualMode && (
            <div className="text-xs text-amber-400 mt-1">
              Viewing {mode.toUpperCase()} history -- account is currently {actualMode.toUpperCase()}
            </div>
          )}
        </div>
        <div className="flex rounded-md border border-neutral-700 overflow-hidden text-sm flex-shrink-0">
          {(['paper', 'live'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setModeOverride(m)}
              className={`px-3 py-1.5 font-medium ${
                mode === m ? 'bg-neutral-800 text-neutral-100' : 'text-neutral-500 hover:text-neutral-300'
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {stats.length > 0 ? (
        <div className="space-y-6">
          <div className="border-b border-neutral-800 pb-5">
            <div className="text-xs font-medium text-neutral-400 mb-3">Win Rate by Strategy &amp; Instrument</div>
            <PerformanceChart stats={stats} />
          </div>

          {stats.map((s) => (
            <div key={s.strategy} className="space-y-2">
              <div className="flex items-baseline gap-4">
                <div className="font-medium text-neutral-200">{s.strategy}</div>
                {DESCRIPTIONS[s.strategy] && (
                  <div className="text-xs text-neutral-600">{DESCRIPTIONS[s.strategy]}</div>
                )}
                <div className="ml-auto flex items-center gap-4 text-xs font-mono text-neutral-500">
                  <span>
                    Win <span className="text-emerald-400">{s.winRate}%</span>
                  </span>
                  <span>{s.trades} trades</span>
                  <span className={s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}
                  </span>
                </div>
              </div>

              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-neutral-600 border-b border-neutral-900">
                    <th className="py-1.5 font-medium">Instrument</th>
                    <th className="py-1.5 font-medium">Win Rate</th>
                    <th className="py-1.5 font-medium">Trades</th>
                    <th className="py-1.5 font-medium">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {s.bySymbol.map((b) => (
                    <tr key={b.symbol} className="border-b border-neutral-900/60">
                      <td className="py-1.5 text-neutral-300">{b.symbol}</td>
                      <td className="py-1.5 font-mono text-emerald-400">{b.winRate}%</td>
                      <td className="py-1.5 font-mono text-neutral-400">{b.trades}</td>
                      <td className={`py-1.5 font-mono ${b.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {b.pnl >= 0 ? '+' : ''}${b.pnl.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-6 text-center text-neutral-600 text-sm">
          {tradesQuery.isFetching ? 'Loading...' : `No closed trades yet in ${mode} mode.`}
        </div>
      )}
    </div>
  )
}

export default function Strategies() {
  const qc = useQueryClient()
  const strategiesQuery = useQuery({ queryKey: ['strategies'], queryFn: api.strategies, refetchInterval: 10000 })

  const update = useMutation({
    mutationFn: ({ name, body }: { name: string; body: { enabled?: boolean; weight?: number } }) =>
      api.updateStrategy(name, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['strategies'] }),
  })

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <div className="text-sm text-neutral-500">
        Weights combine into the ensemble signal a confidence-weighted average. Changes apply on the next trading
        tick.
      </div>
      <div className="rounded-lg border border-neutral-800 bg-neutral-900 divide-y divide-neutral-800">
        {(strategiesQuery.data ?? []).map((s) => (
          <div key={s.name} className="p-4 flex items-center gap-4">
            <button
              onClick={() => update.mutate({ name: s.name, body: { enabled: !s.enabled } })}
              className={`w-11 h-6 rounded-full relative transition-colors flex-shrink-0 ${
                s.enabled ? 'bg-emerald-600' : 'bg-neutral-700'
              }`}
            >
              <span
                className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                  s.enabled ? 'translate-x-5' : 'translate-x-0.5'
                }`}
              />
            </button>

            <div className="flex-1 min-w-0">
              <div className="font-medium text-neutral-200">{s.name}</div>
              <div className="text-xs text-neutral-500">{DESCRIPTIONS[s.name] ?? ''}</div>
            </div>

            <div className="flex items-center gap-2 w-48">
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={s.weight}
                onChange={(e) => update.mutate({ name: s.name, body: { weight: Number(e.target.value) } })}
                className="w-32 accent-emerald-500"
              />
              <span className="font-mono text-sm text-neutral-400 w-10 text-right">{s.weight.toFixed(1)}</span>
            </div>
          </div>
        ))}
        {strategiesQuery.data?.length === 0 && (
          <div className="p-6 text-center text-neutral-600 text-sm">No strategies configured yet</div>
        )}
      </div>

      <PerformanceSection />
    </div>
  )
}
