import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api, type AnalystAnalysis, type AnalystTrade } from '../lib/api'
import { ago, coin, fmtPct, fmtPrice, fmtR, pnlClass } from '../lib/format'
import { ErrorState, Loading, Panel } from './ui'

// The Analyst on the Dashboard: the four major coins, re-analysed every minute,
// its single best opportunity with a confidence score, its open trade and record.

const TONE = {
  BUY: 'text-emerald-300 border-emerald-800 bg-emerald-500/10',
  SELL: 'text-red-300 border-red-900 bg-red-500/10',
  WAIT: 'text-neutral-300 border-neutral-700 bg-neutral-800/60',
} as const

/** Conviction is 0-10; shown as a 0-100% confidence bar. */
export function Confidence({ value, wide = false }: { value: number; wide?: boolean }) {
  const pct = Math.max(0, Math.min(10, value)) * 10
  const tone = value >= 6 ? 'bg-emerald-500' : value >= 4 ? 'bg-amber-500' : 'bg-neutral-500'
  return (
    <div className="flex items-center gap-2">
      <div
        role="meter"
        aria-label="Confidence"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        className={`h-1.5 overflow-hidden rounded-full bg-neutral-800 ${wide ? 'flex-1' : 'w-16'}`}
      >
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs tabular-nums text-neutral-200">{pct}%</span>
    </div>
  )
}

function CoinChip({ a, best }: { a: AnalystAnalysis; best: boolean }) {
  return (
    <div
      className={`min-w-0 rounded-lg border px-2.5 py-2 ${best ? 'border-amber-700/60 bg-amber-500/10' : 'border-neutral-800 bg-neutral-950/40'}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-neutral-100">
          {best && (
            <span className="mr-1 text-amber-400" aria-label="best opportunity">
              ★
            </span>
          )}
          {coin(a.symbol)}
        </span>
        <span className={`rounded border px-1.5 py-0.5 text-[11px] font-semibold ${TONE[a.decision]}`}>{a.decision}</span>
      </div>
      <div className="mt-1.5">
        <Confidence value={a.conviction} wide />
      </div>
    </div>
  )
}

function OpenTrade({ t }: { t: AnalystTrade }) {
  return (
    <div className="rounded-lg border border-sky-900/60 bg-sky-500/5 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <span className={`font-semibold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
            {t.side} {coin(t.symbol)}
          </span>
          {t.live ? (
            <span className="rounded bg-red-600 px-1.5 py-0.5 text-[10px] font-bold text-white">LIVE{t.contracts ? ` x${t.contracts}` : ''}</span>
          ) : (
            <span className="rounded border border-neutral-700 px-1.5 py-0.5 text-[10px] text-neutral-300">paper</span>
          )}
        </span>
        <span className={`font-mono text-sm tabular-nums ${pnlClass(t.r_multiple)}`}>
          {fmtR(t.r_multiple)} <span className="text-xs">({fmtPct(t.pnl_pct)})</span>
        </span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-center">
        <Lvl label="Entry" v={t.entry} cls="text-neutral-100" />
        <Lvl label="Stop" v={t.stop_loss} cls="text-red-300" />
        <Lvl label="TP1" v={t.take_profit} cls="text-emerald-300" />
      </div>
    </div>
  )
}

function Lvl({ label, v, cls }: { label: string; v: number | null; cls: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`truncate font-mono text-[13px] tabular-nums ${cls}`}>{fmtPrice(v)}</div>
    </div>
  )
}

export function AnalystPanel() {
  const q = useQuery({ queryKey: ['analyst'], queryFn: api.analyst, refetchInterval: 30_000 })
  const data = q.data
  const best = data?.analyses.find((a) => a.symbol === data.best) ?? null
  const st = data?.stats
  return (
    <Panel
      title="Analyst"
      right={
        <Link to="/analyst" className="py-1 text-xs font-medium text-sky-400 hover:text-sky-300">
          All readings →
        </Link>
      }
    >
      {q.isError && !data ? (
        <ErrorState what="the Analyst" error={q.error} onRetry={() => q.refetch()} />
      ) : !data ? (
        <Loading className="h-32" label="Loading the Analyst" />
      ) : (
        <div className="grid gap-4 md:grid-cols-[1.1fr_1fr_1fr]">
          <div className="rounded-lg border border-amber-700/50 bg-amber-500/5 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-amber-400">★ Best opportunity</div>
            {best ? (
              <>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-2xl font-semibold text-neutral-100">{coin(best.symbol)}</span>
                  <span className={`text-lg font-semibold ${best.decision === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>
                    {best.decision}
                  </span>
                </div>
                <div className="mt-1">
                  <Confidence value={best.conviction} wide />
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                  <Lvl label="Entry" v={best.entry} cls="text-neutral-100" />
                  <Lvl label="Stop" v={best.stop_loss} cls="text-red-300" />
                  <Lvl label="TP1" v={best.take_profit} cls="text-emerald-300" />
                </div>
              </>
            ) : (
              <div className="mt-2 text-sm text-neutral-300">No coin is strong enough right now -- waiting.</div>
            )}
          </div>

          <div>
            <div className="grid grid-cols-2 gap-2">
              {data.analyses.map((a) => (
                <CoinChip key={a.symbol} a={a} best={a.symbol === data.best} />
              ))}
            </div>
            <div className="mt-2 text-[11px] text-neutral-400">updated {ago(data.updated)}</div>
          </div>

          <div className="space-y-2">
            <div className="text-[11px] uppercase tracking-wide text-neutral-400">
              Open trade{data.open_trades.length > 1 ? `s (${data.open_trades.length})` : ''}
            </div>
            {data.open_trades.length > 0 ? (
              data.open_trades.map((t) => <OpenTrade key={t.id} t={t} />)
            ) : (
              <div className="text-sm text-neutral-300">none open</div>
            )}
            {st && (
              <div className="text-xs text-neutral-300">
                {st.closed} closed · win {st.win_rate != null ? `${Math.round(st.win_rate * 100)}%` : '--'} ·{' '}
                <span className={pnlClass(st.total_r)}>{fmtR(st.total_r)} total</span>
              </div>
            )}
          </div>
        </div>
      )}
    </Panel>
  )
}
