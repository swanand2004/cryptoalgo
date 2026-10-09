import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type AnalystAnalysis, type AnalystFactor, type AnalystTrade } from '../lib/api'
import { formatISTShort } from '../lib/time'
import { coin, fmtPct, fmtPrice, fmtR, fmtSigned, pnlClass } from '../lib/format'
import { Confidence } from '../components/AnalystPanel'
import { Empty, ErrorState, Loading, Panel } from '../components/ui'
import { AiSizingPanel, RiskSettings, StrategyRace, WhyNoTrade } from '../components/AnalystInsights'
import { useIsPhone } from '../lib/hooks'

// The free, rules-based Analyst: every input the system collects, scored the
// same way every time, combined with fixed weights. It paper-trades every call
// and trades the four majors live only when it agrees with the trend book.

const FAMILY_LABEL: Record<AnalystFactor['family'], string> = {
  technical: 'Technical',
  structure: 'Chart, Gann & levels',
  derivatives: 'Derivatives',
  sentiment: 'Sentiment & fundamentals',
  macro: 'Macro',
  live: 'Live order flow (ticks, not scored)',
}
const FAMILY_ORDER: AnalystFactor['family'][] = ['technical', 'structure', 'derivatives', 'sentiment', 'macro']

const DECISION_STYLE = {
  BUY: 'bg-emerald-500/15 text-emerald-300 border-emerald-700',
  SELL: 'bg-red-500/15 text-red-300 border-red-800',
  WAIT: 'bg-neutral-800 text-neutral-400 border-neutral-700',
} as const

const fmt = fmtPrice
const signed = fmtSigned

function ScoreBar({ value }: { value: number }) {
  // -1 .. +1, centred
  const pct = Math.min(1, Math.abs(value)) * 50
  return (
    <div className="relative h-1.5 w-full rounded-full bg-neutral-800">
      <div className="absolute left-1/2 top-0 h-full w-px bg-neutral-600" />
      <div
        className={`absolute top-0 h-full rounded-full ${value >= 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
        style={value >= 0 ? { left: '50%', width: `${pct}%` } : { right: '50%', width: `${pct}%` }}
      />
    </div>
  )
}

export function CoinCard({ a, best }: { a: AnalystAnalysis; best: boolean }) {
  const [open, setOpen] = useState(false)
  const name = coin(a.symbol)
  return (
    <div className={`rounded-lg border bg-neutral-900 p-4 ${best ? 'border-amber-600/70' : 'border-neutral-800'}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          {best && <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-amber-400">★ Best opportunity</div>}
          <div className="text-lg font-semibold text-neutral-100">{name}</div>
          <div className="font-mono text-xs text-neutral-400">{fmt(a.price)}</div>
        </div>
        <div className="text-right">
          <span className={`inline-block rounded-full border px-3 py-1 text-sm font-semibold ${DECISION_STYLE[a.decision]}`}>{a.decision}</span>
          <div className="mt-1 text-[11px] text-neutral-400">score {signed(a.score)}</div>
        </div>
      </div>
      <div className="mt-2">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-neutral-400">Confidence</div>
        <Confidence value={a.conviction} wide />
      </div>

      <div className="mt-3 space-y-1.5">
        {FAMILY_ORDER.filter((f) => a.families[f] !== undefined).map((f) => (
          <div key={f} className="grid grid-cols-[minmax(0,8rem)_1fr_44px] items-center gap-2 text-[11px]">
            <span className="text-neutral-300">{FAMILY_LABEL[f]}</span>
            <ScoreBar value={a.families[f]!} />
            <span className="text-right font-mono text-neutral-300">{signed(a.families[f]!)}</span>
          </div>
        ))}
      </div>

      {a.decision !== 'WAIT' && (
        <div className="mt-3 grid grid-cols-3 gap-2 text-center">
          <div className="rounded border border-neutral-800 px-2 py-1">
            <div className="text-[10px] uppercase tracking-wide text-neutral-400">Entry</div>
            <div className="font-mono text-sm text-neutral-200">{fmt(a.entry)}</div>
          </div>
          <div className="rounded border border-red-900/60 px-2 py-1">
            <div className="text-[10px] uppercase tracking-wide text-neutral-400">Stop loss</div>
            <div className="font-mono text-sm text-red-300">{fmt(a.stop_loss)}</div>
          </div>
          <div className="rounded border border-emerald-900/60 px-2 py-1">
            <div className="text-[10px] uppercase tracking-wide text-neutral-400">TP1</div>
            <div className="font-mono text-sm text-emerald-300">{fmt(a.take_profit)}</div>
          </div>
        </div>
      )}

      {(a.supports?.length || a.resistances?.length) ? (
        <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
          <div>
            <div className="text-neutral-400">Supports</div>
            <div className="font-mono text-emerald-300/80">{(a.supports ?? []).slice(0, 3).map(fmt).join(' · ') || '--'}</div>
          </div>
          <div className="text-right">
            <div className="text-neutral-400">Resistances</div>
            <div className="font-mono text-red-300/80">{(a.resistances ?? []).slice(0, 3).map(fmt).join(' · ') || '--'}</div>
          </div>
        </div>
      ) : null}
      {a.plan_note && a.decision !== 'WAIT' && <div className="mt-1 text-[11px] text-neutral-400">{a.plan_note}</div>}

      {a.notes.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-[11px] text-amber-300/90">
          {a.notes.map((n) => (
            <li key={n}>⚠ {n}</li>
          ))}
        </ul>
      )}

      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="mt-3 py-1 text-xs font-medium text-sky-400 hover:text-sky-300"
      >
        {open ? 'Hide' : 'Show'} the {a.factors.length} readings
      </button>
      {open && (
        <ul className="mt-2 space-y-1.5">
          {a.factors.map((f) => (
            <li key={f.key} className={`text-[11px] ${f.family === 'live' ? 'border-l border-sky-800 pl-2' : ''}`}>
              <div className="flex justify-between gap-2">
                <span className="text-neutral-300">{f.name}</span>
                <span className={`font-mono ${f.score > 0 ? 'text-emerald-400' : f.score < 0 ? 'text-red-400' : 'text-neutral-500'}`}>
                  {signed(f.score)}
                </span>
              </div>
              <div className="text-neutral-400">{f.reading}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

const RESULT_LABEL: Record<string, string> = {
  TP: 'TP1', SL: 'Stop', TRAIL: 'Trailing stop', FLIP: 'Call flipped', SYNC: 'Closed on exchange',
}


function TradeBadges({ t }: { t: AnalystTrade }) {
  return (
    <>
      {t.live && (
        <span className="ml-1.5 rounded bg-red-600 px-1 py-0.5 text-[10px] font-bold text-white">
          LIVE{t.contracts ? ` x${t.contracts}` : ''}
        </span>
      )}
      {t.extended && <span className="ml-1.5 text-[10px] text-amber-300">past TP1</span>}
    </>
  )
}

function TradeCard({ t, open }: { t: AnalystTrade; open: boolean }) {
  return (
    <li className="py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="font-medium text-neutral-100">{coin(t.symbol)}</span>{' '}
          <span className={`text-sm font-semibold ${t.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side}</span>
          <TradeBadges t={t} />
          <div className="text-[11px] text-neutral-400">
            {t.conviction * 10}% confidence · {formatISTShort(t.entry_time)}
          </div>
        </div>
        <div className="text-right">
          <div className={`font-mono text-sm font-semibold tabular-nums ${pnlClass(t.r_multiple ?? t.pnl_pct)}`}>{fmtR(t.r_multiple)}</div>
          <div className="text-[11px] text-neutral-400">
            {open ? fmtPct(t.pnl_pct) : t.result ? (RESULT_LABEL[t.result] ?? t.result) : '--'}
          </div>
        </div>
      </div>
      <div className="mt-1.5 grid grid-cols-3 gap-2 font-mono text-[12px] tabular-nums">
        <span className="truncate text-neutral-300">
          <span className="font-sans text-[10px] text-neutral-400">IN </span>
          {fmt(t.entry)}
        </span>
        <span className="truncate text-red-300">
          <span className="font-sans text-[10px] text-neutral-400">SL </span>
          {fmt(t.stop_loss)}
        </span>
        <span className="truncate text-emerald-300">
          <span className="font-sans text-[10px] text-neutral-400">TP </span>
          {fmt(t.take_profit)}
        </span>
      </div>
      {!open && t.exit_time && (
        <div className="mt-0.5 text-[11px] text-neutral-400">
          exit {fmt(t.exit_price)} · {formatISTShort(t.exit_time)}
        </div>
      )}
    </li>
  )
}

export function TradeTable({ rows, open }: { rows: AnalystTrade[]; open: boolean }) {
  const isPhone = useIsPhone()
  if (rows.length === 0) {
    return <Empty>{open ? 'No open trades' : 'No closed trades yet'}</Empty>
  }
  if (isPhone) {
    return (
      <ul className="divide-y divide-neutral-800">
        {rows.map((t) => (
          <TradeCard key={t.id} t={t} open={open} />
        ))}
      </ul>
    )
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-[13px]">
        <thead>
          <tr className="border-b border-neutral-800 text-left text-[11px] uppercase tracking-wider text-neutral-400">
            <th className="py-2 pr-3 font-medium">Coin</th>
            <th className="py-2 pr-3 font-medium">Side</th>
            <th className="py-2 pr-3 font-medium">Conv.</th>
            <th className="py-2 pr-3 font-medium">Entry time</th>
            <th className="py-2 pr-3 font-medium">Entry</th>
            <th className="py-2 pr-3 font-medium">Stop loss</th>
            <th className="py-2 pr-3 font-medium">TP1</th>
            <th className="py-2 pr-3 font-medium">{open ? 'Now' : 'Exit'}</th>
            <th className="py-2 text-right font-medium">Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t) => (
            <tr key={t.id} className="border-b border-neutral-800/60">
              <td className="py-1.5 pr-3 font-medium text-neutral-100">
                {coin(t.symbol)}
                <TradeBadges t={t} />
              </td>
              <td className={`py-1.5 pr-3 font-medium ${t.side === 'BUY' ? 'text-emerald-400' : 'text-red-400'}`}>{t.side}</td>
              <td className="py-1.5 pr-3 text-neutral-300">{t.conviction}/10</td>
              <td className="py-1.5 pr-3 text-xs text-neutral-400">{formatISTShort(t.entry_time)}</td>
              <td className="py-1.5 pr-3 font-mono tabular-nums text-neutral-200">{fmt(t.entry)}</td>
              <td className="py-1.5 pr-3 font-mono tabular-nums text-red-300">{fmt(t.stop_loss)}</td>
              <td className="py-1.5 pr-3 font-mono tabular-nums text-emerald-300">{fmt(t.take_profit)}</td>
              <td className="py-1.5 pr-3 font-mono tabular-nums text-neutral-300">
                {open ? fmtPct(t.pnl_pct) : <span className="font-sans text-xs text-neutral-400">{formatISTShort(t.exit_time)}</span>}
              </td>
              <td className={`py-1.5 text-right font-mono tabular-nums ${pnlClass(t.r_multiple)}`}>
                {t.result ? `${RESULT_LABEL[t.result] ?? t.result} · ` : ''}
                {fmtR(t.r_multiple)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Analyst() {
  const q = useQuery({ queryKey: ['analyst'], queryFn: api.analyst, refetchInterval: 60_000 })
  const data = q.data
  const st = data?.stats
  return (
    <div className="mx-auto max-w-7xl space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-xl font-semibold text-neutral-100">Analyst</h1>
        <div className="text-xs text-neutral-400">updated {data?.updated ? formatISTShort(data.updated) : '--'}</div>
      </div>

      <details className="rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-[13px] text-neutral-300">
        <summary className="cursor-pointer py-1 text-sm font-medium text-neutral-200">How the Analyst works</summary>
        <p className="mt-2 leading-relaxed">
          A rules-based analyst: technicals, derivatives, sentiment &amp; fundamentals and macro, scored the same way every
          minute on the four major coins (BTC, ETH, SOL, BNB), with live order flow from tick data. It works WITH the trend
          book: it reads trend, momentum, chart patterns, Gann, support/resistance levels, derivatives, sentiment and macro, and
          trades the single <span className="text-amber-300">best opportunity</span> at 40%+ confidence only in the trend
          book&rsquo;s direction for that coin, one trade at a time. The stop trails behind the best price and never moves back;
          at TP1 it takes profit unless the call is still strong, then locks +0.5R and lets it run.
        </p>
      </details>

      {q.isError && <ErrorState what="the Analyst" error={q.error} onRetry={() => q.refetch()} />}

      {st && (
        <div className="grid grid-cols-3 gap-2 rounded-xl border border-neutral-800 bg-neutral-900 p-3 text-center sm:grid-cols-5">
          <Stat label="Closed" value={String(st.closed)} />
          <Stat label="Win rate" value={st.win_rate != null ? `${Math.round(st.win_rate * 100)}%` : '--'} />
          <Stat label="Avg" value={fmtR(st.avg_r)} cls={pnlClass(st.avg_r)} />
          <Stat label="Total" value={fmtR(st.total_r)} cls={pnlClass(st.total_r)} />
          <Stat label="Open" value={String(data?.open_trades.length ?? 0)} />
        </div>
      )}

      {data && data.analyses.length > 0 && <WhyNoTrade data={data} />}
      {data && <RiskSettings rs={data.risk_settings} />}

      {q.isLoading ? (
        <Loading className="h-48" label="Loading the Analyst" />
      ) : data && data.analyses.length === 0 ? (
        <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-6 text-center text-sm text-neutral-300">
          The first analysis runs a few minutes after the server starts, then every minute.
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {data?.analyses.map((a) => <CoinCard key={a.symbol} a={a} best={a.symbol === data.best} />)}
        </div>
      )}

      {data && <StrategyRace race={data.variants} />}

      <AiSizingPanel />

      {data && (
        <>
          <Panel title="Open trades" bodyClassName="px-4 py-2">
            <TradeTable rows={data.open_trades} open />
          </Panel>
          <Panel title="Closed trades" bodyClassName="px-4 py-2">
            <TradeTable rows={data.closed_trades} open={false} />
          </Panel>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, cls = 'text-neutral-100' }: { label: string; value: string; cls?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`font-mono text-base font-semibold tabular-nums ${cls}`}>{value}</div>
    </div>
  )
}
