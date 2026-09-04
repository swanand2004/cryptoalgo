import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, netPnl, type ClosedTradeRow } from '../lib/api'

// WIN RATE ALONE IS NOT A RESULT, and this page used to present it as one --
// a bar whose length was the win rate, with 100% as the implied goal.
//
// The live strategies here have run at a 58-60% win rate while losing money,
// because their average winner was 0.63x their average loser. At that payoff
// the break-even win rate is 61%, so 59% is a LOSING number that looks like a
// good one. A page that draws 59% as a long green bar is actively misleading
// about the thing it exists to report.
//
// So every strategy is now measured against ITS OWN break-even win rate,
// derived from its own payoff ratio, and the bar shows the gap. Expectancy per
// trade sits next to it, because that is the number that actually decides
// whether more trades help or hurt.

const DESCRIPTIONS: Record<string, string> = {
  trend_momentum: 'EMA cross + MACD confirmation, ATR-normalized. Rides trends.',
  graha:
    'EXPERIMENTAL, deep Panchang mechanism. Real ephemeris math (JPL DE421 + Lahiri ayanamsa + Meeus lunar nodes) covering all 9 Navagraha: gated flat during Rahu Kaal, Vishti Karana, or a severe Nitya Yoga; tilted by lunar paksha, nakshatra-lord benefic/malefic, and planetary aspects to the Moon; discounted during Mercury/Mars/Saturn retrograde. No established causal link between astrology and markets -- kept enabled as an explicit, informed override, judged purely on backtest results.',
  tsmom:
    'Time-series momentum: a barbell of a short + long lookback horizon, each voting on the sign of its own trailing return (vol-normalized). Grounded in Moskowitz/Ooi/Pedersen (2012) and AQR’s "Century of Evidence on Trend-Following."',
}

const SYMBOL_COLORS: Record<string, string> = {
  BTCUSD: '#3987e5',
  ETHUSD: '#d95926',
  SOLUSD: '#199e70',
}
const SYMBOL_ORDER = ['BTCUSD', 'ETHUSD', 'SOLUSD']

interface Stats {
  trades: number
  wins: number
  winRate: number
  pnl: number
  avgWin: number
  avgLoss: number
  payoff: number | null
  breakevenWinRate: number | null
  expectancy: number
  profitFactor: number | null
  /** 2-sigma band on the win-rate estimate, in percentage points. */
  band: number
}

interface SymbolStats extends Stats {
  symbol: string
}
interface StrategyStats extends Stats {
  strategy: string
  bySymbol: SymbolStats[]
}

function summarize(rows: ClosedTradeRow[]): Stats {
  // Wins, losses and PnL are all NET of fees -- a trade that made money on
  // price but lost it to commission is a loss, not a win.
  const pnls = rows.map(netPnl)
  const winners = pnls.filter((p) => p > 0)
  const losers = pnls.filter((p) => p <= 0)
  const n = rows.length
  const grossWin = winners.reduce((a, b) => a + b, 0)
  const grossLoss = Math.abs(losers.reduce((a, b) => a + b, 0))
  const avgWin = winners.length ? grossWin / winners.length : 0
  const avgLoss = losers.length ? grossLoss / losers.length : 0
  // Undefined rather than infinite when nothing has lost yet: "no losses in 4
  // trades" is not a payoff of infinity, it is not enough trades.
  const payoff = avgLoss > 0 && winners.length ? avgWin / avgLoss : null
  const winRate = n ? (winners.length / n) * 100 : 0
  return {
    trades: n,
    wins: winners.length,
    winRate,
    pnl: pnls.reduce((a, b) => a + b, 0),
    avgWin,
    avgLoss,
    payoff,
    // Solve winRate * avgWin = (1 - winRate) * avgLoss  ->  1 / (1 + payoff).
    breakevenWinRate: payoff != null ? (1 / (1 + payoff)) * 100 : null,
    expectancy: n ? pnls.reduce((a, b) => a + b, 0) / n : 0,
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : null,
    band: n ? (100 / Math.sqrt(n)) : 0,
  }
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
  return Array.from(byStrategy.entries())
    .map(([strategy, map]) => ({
      strategy,
      ...summarize(Array.from(map.values()).flat()),
      bySymbol: Array.from(map.entries())
        .map(([symbol, rows]) => ({ symbol, ...summarize(rows) }))
        // Fixed symbol order, not trade-count order: rows must not reshuffle
        // under the cursor every time a refetch changes which symbol has the
        // most trades.
        .sort((a, b) => SYMBOL_ORDER.indexOf(a.symbol) - SYMBOL_ORDER.indexOf(b.symbol)),
    }))
    .sort((a, b) => b.trades - a.trades)
}

const money = (v: number) => `${v >= 0 ? '+' : '−'}$${Math.abs(v).toFixed(2)}`

/** Win rate drawn against the win rate this strategy actually NEEDS.
 *
 * The scale is 0-100% with a marker at break-even, so "above the line" and
 * "below the line" are the readable states rather than "long bar" and "short
 * bar". Colour follows the gap, not the raw rate: 59% is red when break-even
 * is 61%, which is exactly the case this page previously drew as success. */
function WinRateBar({ s }: { s: Stats }) {
  if (!s.trades) return <div className="h-6" />
  const be = s.breakevenWinRate
  const edge = be != null ? s.winRate - be : null
  const clear = edge != null && Math.abs(edge) > s.band
  const color = edge == null ? '#8b8578' : !clear ? '#8b8578' : edge > 0 ? '#4bb98a' : '#e2645f'

  return (
    <div className="w-full">
      <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-neutral-900 ring-1 ring-inset ring-neutral-800">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${Math.max(s.winRate, 1)}%`, background: color, opacity: 0.85 }}
        />
        {be != null && (
          <>
            {/* the uncertainty on the estimate, so a 3-trade sample cannot
                look like a verdict */}
            <div
              className="absolute inset-y-0 bg-neutral-100/10"
              style={{
                left: `${Math.max(0, s.winRate - s.band)}%`,
                width: `${Math.min(100, s.band * 2)}%`,
              }}
            />
            <div className="absolute -top-0.5 h-3.5 w-[2px] bg-neutral-100" style={{ left: `${be}%` }} title={`break-even ${be.toFixed(1)}%`} />
          </>
        )}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[9px] tabular-nums text-neutral-600">
        <span style={{ color }}>{s.winRate.toFixed(1)}% win</span>
        <span>{be != null ? `needs ${be.toFixed(1)}%` : 'no losses yet'}</span>
      </div>
    </div>
  )
}

function Metric({ label, value, tone, hint }: { label: string; value: string; tone?: string; hint?: string }) {
  return (
    <div title={hint}>
      <div className="text-[9px] uppercase tracking-[0.12em] text-neutral-600">{label}</div>
      <div className={`font-mono text-sm tabular-nums ${tone ?? 'text-neutral-300'}`}>{value}</div>
    </div>
  )
}

function StrategyCard({ s }: { s: StrategyStats }) {
  const edge = s.breakevenWinRate != null ? s.winRate - s.breakevenWinRate : null
  const clear = edge != null && Math.abs(edge) > s.band
  const verdict =
    s.trades < 30
      ? { label: 'too few trades', cls: 'bg-neutral-800 text-neutral-400' }
      : edge == null || !clear
      ? { label: 'inconclusive', cls: 'bg-neutral-800 text-neutral-400' }
      : edge > 0
      ? { label: 'above break-even', cls: 'bg-emerald-500/15 text-emerald-400' }
      : { label: 'below break-even', cls: 'bg-red-500/15 text-red-400' }

  return (
    <div className="rounded-lg border border-neutral-800/80 bg-neutral-950/40 p-4">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-medium text-neutral-100">{s.strategy}</span>
        <span className={`rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider ${verdict.cls}`}>
          {verdict.label}
        </span>
        <span className={`ml-auto font-mono text-sm tabular-nums ${s.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {money(s.pnl)}
        </span>
      </div>

      <div className="mt-3 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <WinRateBar s={s} />
        <div className="grid grid-cols-4 gap-3">
          <Metric
            label="Expectancy"
            value={s.trades ? money(s.expectancy) : '--'}
            tone={s.expectancy >= 0 ? 'text-emerald-400' : 'text-red-400'}
            hint="Average net PnL per trade. Negative means every additional trade loses money on average, whatever the win rate says."
          />
          <Metric
            label="Payoff"
            value={s.payoff != null ? `${s.payoff.toFixed(2)}x` : '--'}
            hint="Average winner divided by average loser. This is what sets the break-even win rate."
          />
          <Metric
            label="Profit factor"
            value={s.profitFactor != null ? s.profitFactor.toFixed(2) : '--'}
            tone={s.profitFactor != null && s.profitFactor >= 1 ? 'text-emerald-400' : 'text-red-400'}
            hint="Gross winnings divided by gross losses. Below 1.0 is a losing strategy."
          />
          <Metric label="Trades" value={String(s.trades)} hint={`2-sigma band on the win rate: ±${s.band.toFixed(1)} points`} />
        </div>
      </div>

      {s.bySymbol.length > 0 && (
        <div className="mt-3 border-t border-neutral-800/60 pt-3">
          <div className="space-y-1.5">
            {s.bySymbol.map((b) => (
              <div key={b.symbol} className="flex items-center gap-3 text-[11px]">
                <span className="flex w-20 shrink-0 items-center gap-1.5 text-neutral-400">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: SYMBOL_COLORS[b.symbol] ?? '#666' }} />
                  {b.symbol}
                </span>
                <span className="w-24 shrink-0 font-mono tabular-nums text-neutral-500">
                  {b.winRate.toFixed(0)}%
                  {b.breakevenWinRate != null && (
                    <span className="text-neutral-700"> / {b.breakevenWinRate.toFixed(0)}%</span>
                  )}
                </span>
                <span className="w-16 shrink-0 font-mono tabular-nums text-neutral-600">{b.trades}t</span>
                <span
                  className={`w-20 shrink-0 font-mono tabular-nums ${b.expectancy >= 0 ? 'text-emerald-500/80' : 'text-red-500/80'}`}
                  title="expectancy per trade"
                >
                  {money(b.expectancy)}
                </span>
                <span className={`ml-auto font-mono tabular-nums ${b.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {money(b.pnl)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-2 text-[9px] text-neutral-700">win % / break-even % &middot; trades &middot; expectancy &middot; net PnL</div>
        </div>
      )}

      {DESCRIPTIONS[s.strategy] && (
        <p className="mt-3 border-t border-neutral-800/60 pt-2.5 text-[10px] leading-relaxed text-neutral-600">
          {DESCRIPTIONS[s.strategy]}
        </p>
      )}
    </div>
  )
}

function PerformanceSection({ running }: { running: string[] }) {
  // Default to the account's ACTUAL current mode rather than a hardcoded
  // "paper", so this never silently shows wrong-mode performance. A manual
  // pick still sticks.
  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })
  const [modeOverride, setModeOverride] = useState<'paper' | 'live' | null>(null)
  const actualMode = modeQuery.data?.mode ?? 'paper'
  const mode = modeOverride ?? (actualMode as 'paper' | 'live')
  const tradesQuery = useQuery({
    queryKey: ['closed_trades_performance', mode],
    queryFn: () => api.closedTrades(mode),
    refetchInterval: 15000,
  })

  const all = groupPerformance(tradesQuery.data ?? [])

  // Only the strategies actually in the live lineup get a card. Six retired
  // ones (mean_reversion, ml_signal, orderbook_imbalance, rahu, funding_carry,
  // ensemble) still appear in closed-trade history and used to be rendered as
  // full cards, which made the page look like it was running nine strategies.
  //
  // Their PnL is NOT discarded, though: it is real money this paper account
  // made and lost, so dropping it silently would leave the total on this page
  // disagreeing with the account. It is summarised in one line instead.
  const runningSet = new Set(running)
  const stats = all.filter((s) => runningSet.has(s.strategy))
  const retired = all.filter((s) => !runningSet.has(s.strategy))
  const retiredTotal = retired.reduce(
    (acc, s) => ({ trades: acc.trades + s.trades, pnl: acc.pnl + s.pnl }),
    { trades: 0, pnl: 0 }
  )
  const total = stats.reduce(
    (acc, s) => ({ trades: acc.trades + s.trades, pnl: acc.pnl + s.pnl }),
    { trades: 0, pnl: 0 }
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-neutral-200">Realised performance</h2>
          <p className="mt-0.5 text-[11px] text-neutral-600">
            This account&rsquo;s actual closed trades for the {running.length} running strategies &mdash; real fills,
            net of fees. Not a backtest.
          </p>
          {mode !== actualMode && (
            <p className="mt-1 text-[11px] text-amber-400">
              Viewing {mode.toUpperCase()} history &mdash; account is currently {actualMode.toUpperCase()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {total.trades > 0 && (
            <span className={`font-mono text-sm tabular-nums ${total.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {money(total.pnl)}
              <span className="ml-1.5 text-[10px] text-neutral-600">{total.trades} trades</span>
            </span>
          )}
          <div className="flex overflow-hidden rounded-md border border-neutral-800 text-xs">
            {(['paper', 'live'] as const).map((m) => (
              <button
                key={m}
                type="button"
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
      </div>

      {stats.length > 0 ? (
        <div className="space-y-3">
          {stats.map((s) => (
            <StrategyCard key={s.strategy} s={s} />
          ))}
          {retired.length > 0 && (
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 rounded-lg border border-neutral-800/60 bg-neutral-950/30 px-3 py-2 text-[10px] text-neutral-600">
              <span className="uppercase tracking-[0.12em] text-neutral-500">Retired</span>
              <span>
                {retired.length} strategies no longer in the lineup hold{' '}
                <span className="font-mono tabular-nums text-neutral-500">{retiredTotal.trades}</span> historical
                trades worth{' '}
                <span className={`font-mono tabular-nums ${retiredTotal.pnl >= 0 ? 'text-emerald-500/80' : 'text-red-500/80'}`}>
                  {money(retiredTotal.pnl)}
                </span>
                . Kept in the total so this page reconciles with the account, not shown as cards.
              </span>
              <span className="ml-auto font-mono text-neutral-700">{retired.map((r) => r.strategy).join(' · ')}</span>
            </div>
          )}

          <p className="text-[10px] leading-relaxed text-neutral-600">
            Break-even win rate is <span className="font-mono text-neutral-500">1 / (1 + payoff)</span>. A strategy
            winning 59% of the time with an average winner 0.63x its average loser needs 61% and is losing money &mdash;
            which is the live lineup&rsquo;s actual history, and why win rate is drawn against that line rather than
            against 100%. The shaded band is the 2-sigma uncertainty on the estimate, so a short run cannot read as a
            verdict.
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-neutral-800/80 bg-neutral-950/40 py-10 text-center text-sm text-neutral-600">
          {tradesQuery.isFetching
            ? 'Loading…'
            : retired.length > 0
            ? `No closed trades yet from the running strategies in ${mode} mode (${retiredTotal.trades} from retired ones).`
            : `No closed trades yet in ${mode} mode.`}
        </div>
      )}
    </div>
  )
}

/** Weight slider that commits ONE request when you let go.
 *
 * It previously fired a PUT on every onChange, so a single drag across the
 * track sent a request per step and raced its own invalidations -- the value
 * could visibly snap backwards mid-drag as an earlier response landed after a
 * later one. */
function WeightSlider({
  value,
  disabled,
  onCommit,
}: {
  value: number
  disabled?: boolean
  onCommit: (v: number) => void
}) {
  const [local, setLocal] = useState(value)
  useEffect(() => setLocal(value), [value])
  const dirty = Math.abs(local - value) > 1e-9

  return (
    <div className="flex w-44 items-center gap-2">
      <input
        type="range"
        min={0}
        max={2}
        step={0.1}
        value={local}
        disabled={disabled}
        aria-label="strategy weight"
        onChange={(e) => setLocal(Number(e.target.value))}
        onPointerUp={() => dirty && onCommit(local)}
        onKeyUp={() => dirty && onCommit(local)}
        onBlur={() => dirty && onCommit(local)}
        className="w-28 accent-amber-500 disabled:opacity-40"
      />
      <span className={`w-10 text-right font-mono text-sm tabular-nums ${dirty ? 'text-amber-400' : 'text-neutral-400'}`}>
        {local.toFixed(1)}
      </span>
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

  const rows = strategiesQuery.data ?? []
  const enabled = rows.filter((s) => s.enabled)
  const totalWeight = enabled.reduce((a, s) => a + s.weight, 0)

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/40">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-neutral-800/60 px-4 py-3">
          <div>
            <h1 className="text-sm font-medium text-neutral-200">Strategy weights</h1>
            <p className="mt-0.5 text-[11px] text-neutral-600">
              Weights combine into a confidence-weighted ensemble signal. Changes apply on the next trading tick.
            </p>
          </div>
          <div className="text-right">
            <div className="font-mono text-sm tabular-nums text-neutral-300">
              {enabled.length}
              <span className="text-neutral-600">/{rows.length} on</span>
            </div>
            <div className="font-mono text-[10px] tabular-nums text-neutral-600">
              total weight {totalWeight.toFixed(1)}
            </div>
          </div>
        </div>

        <div className="divide-y divide-neutral-800/60">
          {rows.map((s) => (
            <div key={s.name} className="flex items-center gap-4 px-4 py-3">
              <button
                type="button"
                role="switch"
                aria-checked={s.enabled}
                aria-label={`${s.enabled ? 'Disable' : 'Enable'} ${s.name}`}
                onClick={() => update.mutate({ name: s.name, body: { enabled: !s.enabled } })}
                className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${
                  s.enabled ? 'bg-emerald-600' : 'bg-neutral-700'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                    s.enabled ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>

              <div className="min-w-0 flex-1">
                <div className={`font-medium ${s.enabled ? 'text-neutral-100' : 'text-neutral-500'}`}>{s.name}</div>
                {DESCRIPTIONS[s.name] && (
                  <div className="line-clamp-2 text-[11px] text-neutral-600">{DESCRIPTIONS[s.name]}</div>
                )}
              </div>

              <WeightSlider
                value={s.weight}
                disabled={!s.enabled}
                onCommit={(weight) => update.mutate({ name: s.name, body: { weight } })}
              />
            </div>
          ))}
          {rows.length === 0 && (
            <div className="py-10 text-center text-sm text-neutral-600">
              {strategiesQuery.isLoading ? 'Loading…' : 'No strategies configured yet'}
            </div>
          )}
        </div>
      </div>

      <PerformanceSection running={rows.map((r) => r.name)} />
    </div>
  )
}
