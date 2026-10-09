import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { optionsApi, type OptionQuote, type S7Row, type Underlying } from '../lib/optionsApi'
import { formatISTShort } from '../lib/time'

// Delta BTC/ETH options: a read-only chain from Delta's public API, and the
// PAPER "S7" tracker. Nothing on this page can place an order.

function Panel({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900/50">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800/70 px-4 py-2.5">
        <h2 className="text-[11px] font-medium uppercase tracking-[0.16em] text-neutral-400">{title}</h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  )
}

function num(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return '--'
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? '--' : `${(v * 100).toFixed(digits)}%`
}

function usd(v: number | null | undefined, digits = 4): string {
  if (v == null) return '--'
  return `${v < 0 ? '-' : ''}$${Math.abs(v).toFixed(digits)}`
}

function signed(v: number | null | undefined, digits = 2): string {
  if (v == null) return '--'
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`
}

const SIDE_COLS = ['Bid', 'Ask', 'Mark', 'IV', 'Δ', 'Γ', 'Θ', 'Vega', 'OI'] as const

function sideCells(q: OptionQuote | null, itm: boolean) {
  const base = `py-1 px-1.5 text-right font-mono ${itm ? 'bg-neutral-800/40' : ''}`
  if (!q) return SIDE_COLS.map((c) => <td key={c} className={`${base} text-neutral-700`}>--</td>)
  return [
    <td key="bid" className={`${base} text-emerald-300/90`}>{num(q.bid, 1)}</td>,
    <td key="ask" className={`${base} text-red-300/90`}>{num(q.ask, 1)}</td>,
    <td key="mark" className={`${base} text-neutral-200`}>{num(q.mark, 1)}</td>,
    <td key="iv" className={`${base} text-neutral-400`}>{pct(q.iv)}</td>,
    <td key="d" className={`${base} text-neutral-300`}>{num(q.delta, 2)}</td>,
    <td key="g" className={`${base} text-neutral-500`}>{q.gamma == null ? '--' : q.gamma.toExponential(1)}</td>,
    <td key="t" className={`${base} text-neutral-500`}>{num(q.theta, 1)}</td>,
    <td key="v" className={`${base} text-neutral-500`}>{num(q.vega, 1)}</td>,
    <td key="oi" className={`${base} text-neutral-400`}>{q.oi_contracts == null ? '--' : q.oi_contracts.toLocaleString()}</td>,
  ]
}

function ChainPanel() {
  const [underlying, setUnderlying] = useState<Underlying>('BTC')
  const [expiry, setExpiry] = useState<string | null>(null)
  const { data, isLoading, error } = useQuery({
    queryKey: ['options-chain', underlying, expiry],
    queryFn: () => optionsApi.chain(underlying, expiry),
    refetchInterval: 60_000,
  })
  const spot = data?.spot ?? null
  const atm =
    spot != null && data && data.strikes.length
      ? data.strikes.reduce((best, s) => (Math.abs(s.strike - spot) < Math.abs(best - spot) ? s.strike : best), data.strikes[0].strike)
      : null

  const right = (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex overflow-hidden rounded-md border border-neutral-700 text-xs">
        {(['BTC', 'ETH'] as Underlying[]).map((u) => (
          <button
            key={u}
            type="button"
            onClick={() => {
              setUnderlying(u)
              setExpiry(null)
            }}
            className={`px-3 py-1 font-medium ${u === underlying ? 'bg-neutral-700 text-neutral-50' : 'bg-neutral-900 text-neutral-400 hover:text-neutral-200'}`}
          >
            {u}
          </button>
        ))}
      </div>
      <select
        value={data?.expiry ?? ''}
        onChange={(e) => setExpiry(e.target.value || null)}
        className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
      >
        {(data?.expiries ?? []).map((e) => (
          <option key={e.code} value={e.code}>
            {formatISTShort(e.expiry)} · {e.days < 1 ? `${Math.round(e.days * 24)}h` : `${e.days.toFixed(1)}d`}
          </option>
        ))}
      </select>
    </div>
  )

  return (
    <Panel title="Option chain · Delta Exchange (public data)" right={right}>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-neutral-500">
        <span>
          spot <span className="font-mono text-neutral-200">{num(spot, 2)}</span>
        </span>
        <span>
          1 contract = <span className="font-mono text-neutral-300">{data?.contract_value ?? '--'} {underlying}</span>
        </span>
        <span>prices per 1 {underlying}, USD</span>
        <span>updated {data?.fetched_at ? formatISTShort(data.fetched_at) : '--'}</span>
      </div>
      {isLoading ? (
        <div className="h-64 animate-pulse rounded-lg bg-neutral-900" />
      ) : error ? (
        <div className="py-6 text-center text-sm text-red-400">{(error as Error).message}</div>
      ) : !data || data.strikes.length === 0 ? (
        <div className="py-6 text-center text-sm text-neutral-500">No listed options for this expiry.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-[11px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-[0.12em]">
                <th colSpan={SIDE_COLS.length} className="pb-1 text-center font-medium text-emerald-400/80">Calls</th>
                <th />
                <th colSpan={SIDE_COLS.length} className="pb-1 text-center font-medium text-red-400/80">Puts</th>
              </tr>
              <tr className="border-b border-neutral-800 text-[10px] uppercase tracking-[0.1em] text-neutral-500">
                {[...SIDE_COLS].reverse().map((c) => (
                  <th key={`c-${c}`} className="px-1.5 py-1.5 text-right font-medium">{c}</th>
                ))}
                <th className="px-2 py-1.5 text-center font-medium text-neutral-300">Strike</th>
                {SIDE_COLS.map((c) => (
                  <th key={`p-${c}`} className="px-1.5 py-1.5 text-right font-medium">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.strikes.map((s) => {
                const callItm = spot != null && s.strike < spot
                const putItm = spot != null && s.strike > spot
                return (
                  <tr key={s.strike} className={`border-b border-neutral-900 ${s.strike === atm ? 'outline outline-1 outline-amber-700/60' : ''}`}>
                    {sideCells(s.call, callItm).reverse()}
                    <td className="px-2 py-1 text-center font-mono font-semibold text-neutral-100">{s.strike.toLocaleString()}</td>
                    {sideCells(s.put, putItm)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-[10px] text-neutral-600">
        Shaded = in the money. Amber outline = strike nearest spot. Θ and vega are USD per 1 {underlying} per day / per IV point.
        Cached ~60 s. Read-only: this page never sends orders.
      </p>
    </Panel>
  )
}

const DECISION_STYLE: Record<S7Row['decision'], string> = {
  CALL: 'text-emerald-400',
  PUT: 'text-red-400',
  NONE: 'text-neutral-500',
}

function S7Table({ rows }: { rows: S7Row[] }) {
  if (rows.length === 0) {
    return <div className="py-4 text-center text-sm text-neutral-600">No weeks recorded yet. The first entry is the next Friday 12:00 UTC.</div>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] text-[12px]">
        <thead>
          <tr className="border-b border-neutral-800 text-left text-[10px] uppercase tracking-[0.12em] text-neutral-500">
            <th className="py-2 pr-3 font-medium">Week</th>
            <th className="py-2 pr-3 font-medium">Coin</th>
            <th className="py-2 pr-3 font-medium">Signal</th>
            <th className="py-2 pr-3 font-medium">Decision</th>
            <th className="py-2 pr-3 font-medium">Contract</th>
            <th className="py-2 pr-3 font-medium">Δ / IV</th>
            <th className="py-2 pr-3 font-medium">Paid</th>
            <th className="py-2 pr-3 font-medium">At expiry</th>
            <th className="py-2 text-right font-medium">P&amp;L</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-neutral-900 align-top">
              <td className="py-1.5 pr-3 font-mono text-neutral-400">{r.week}</td>
              <td className="py-1.5 pr-3 font-medium text-neutral-200">{r.underlying}</td>
              <td className="py-1.5 pr-3 font-mono text-neutral-300">
                {signed(r.signal, 3)}
                {r.signal_source && <span className="ml-1 text-[10px] text-neutral-600">{r.signal_source}</span>}
              </td>
              <td className={`py-1.5 pr-3 font-semibold ${DECISION_STYLE[r.decision]}`}>
                {r.decision}
                {r.note && <div className="text-[10px] font-normal text-neutral-600">{r.note}</div>}
              </td>
              <td className="py-1.5 pr-3 font-mono text-neutral-300">
                {r.symbol ?? '--'}
                {r.expiry && <div className="text-[10px] text-neutral-600">exp {formatISTShort(r.expiry)}</div>}
              </td>
              <td className="py-1.5 pr-3 font-mono text-neutral-400">
                {r.delta_at_entry == null ? '--' : `${num(r.delta_at_entry, 2)} / ${pct(r.iv_at_entry, 0)}`}
              </td>
              <td className="py-1.5 pr-3 font-mono text-neutral-300">
                {r.premium == null ? '--' : (
                  <>
                    {num(r.premium, 1)} <span className="text-[10px] text-neutral-600">{r.premium_source}</span>
                    <div className="text-[10px] text-neutral-600">{usd(r.cost_usd)} incl. fee {usd(r.entry_fee)}</div>
                  </>
                )}
              </td>
              <td className="py-1.5 pr-3 font-mono text-neutral-400">
                {r.decision === 'NONE' ? '--' : r.is_open ? (
                  <span className="text-amber-300/80">open</span>
                ) : (
                  <>
                    {num(r.underlying_at_expiry, 1)}
                    <div className="text-[10px] text-neutral-600">
                      intrinsic {num(r.settle_value, 1)} · {r.settle_source}
                    </div>
                  </>
                )}
              </td>
              <td className={`py-1.5 text-right font-mono ${(r.pnl_usd ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {r.pnl_usd == null ? '--' : (
                  <>
                    {usd(r.pnl_usd)}
                    <div className="text-[10px]">{r.pnl_pct == null ? '' : `${signed(r.pnl_pct * 100, 0)}%`}</div>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function S7Panel() {
  const { data, isLoading } = useQuery({ queryKey: ['options-s7'], queryFn: optionsApi.s7, refetchInterval: 60_000 })
  const st = data?.stats
  return (
    <Panel
      title="S7 paper tracker"
      right={<span className="text-[11px] text-neutral-500">next entry {data ? formatISTShort(data.next_entry) : '--'}</span>}
    >
      <ul className="mb-3 space-y-0.5 text-[12px] text-neutral-400">
        <li>Every Friday 12:00 UTC, for BTC and ETH, read the live trend-book signal.</li>
        <li>
          Signal &gt; +0.2: paper-buy the ~7-day call nearest 0.40 delta. Signal &lt; −0.2: the put nearest −0.40 delta.
          Otherwise nothing.
        </li>
        <li>1 contract (0.001 BTC / 0.01 ETH) at the ask (mark if no ask). Fee min(0.01% notional, 3.5% premium) × 1.18, on entry and on settlement if in the money.</li>
        <li>Held to expiry and settled at intrinsic value. These rules are fixed in advance.</li>
      </ul>
      {st && (
        <div className="mb-3 flex flex-wrap gap-x-6 gap-y-1 rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2.5 text-[12px] text-neutral-400">
          <span><span className="text-neutral-200">{st.weeks}</span> weeks</span>
          <span><span className="text-neutral-200">{st.trades}</span> trades</span>
          <span>open {st.open}</span>
          <span>settled {st.settled}</span>
          <span>win {st.settled ? `${Math.round((st.wins / st.settled) * 100)}%` : '--'}</span>
          <span className={st.total_pnl_usd >= 0 ? 'text-emerald-400' : 'text-red-400'}>total {usd(st.total_pnl_usd)}</span>
          <span>on premium {st.return_on_premium == null ? '--' : `${signed(st.return_on_premium * 100, 1)}%`}</span>
        </div>
      )}
      {isLoading ? <div className="h-32 animate-pulse rounded-lg bg-neutral-900" /> : <S7Table rows={data?.rows ?? []} />}
    </Panel>
  )
}

export default function Options() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-neutral-100">Options</h1>
        <p className="text-[12px] text-neutral-500">BTC and ETH options on Delta Exchange: live chain and the S7 forward test.</p>
      </div>
      <div className="rounded-lg border border-amber-800/60 bg-amber-950/30 px-4 py-3 text-[13px] text-amber-200">
        Paper only. Our backtest found no options strategy that beats costs; this tracks S7 forward before any real money.
      </div>
      <S7Panel />
      <ChainPanel />
    </div>
  )
}
