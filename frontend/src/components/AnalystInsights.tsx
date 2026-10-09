import { useQuery } from '@tanstack/react-query'
import {
  api,
  type AnalystAnalysis,
  type AnalystRiskSettings,
  type AnalystSnapshot,
  type AnalystVariantRace,
  type BookFilterRecord,
} from '../lib/api'
import { coin, fmtR, fmtSigned, pnlClass } from '../lib/format'
import { formatISTShort } from '../lib/time'
import { Empty, ErrorState, Loading, Panel } from './ui'

// Three read-only panels for the Analyst page: why each coin did not trade on
// the latest run, the risk settings in force, and the ML book filter's daily
// sizing multipliers (shadow = logged only).

const BOOK_LINE = 0.15 // agrees_with_book: the trend book "holds" a coin at |signal| >= 0.15
const MAJORS = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD']

type Tone = 'ok' | 'wait' | 'block' | 'stop'
const TONE: Record<Tone, { dot: string; text: string }> = {
  ok: { dot: 'bg-emerald-400', text: 'text-emerald-300' },
  wait: { dot: 'bg-neutral-500', text: 'text-neutral-300' },
  block: { dot: 'bg-amber-400', text: 'text-amber-200' },
  stop: { dot: 'bg-red-500', text: 'text-red-300' },
}

const pct = (v: number | null | undefined, digits = 0) =>
  v == null || !Number.isFinite(v) ? '--' : `${(v * 100).toFixed(digits)}%`

function bookWords(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return 'has no signal'
  if (s >= BOOK_LINE) return `is long (${fmtSigned(s)})`
  if (s <= -BOOK_LINE) return `is short (${fmtSigned(s)})`
  return `is flat (${fmtSigned(s)})`
}

/** One coin's latest gate, in plain words. */
export function gateText(
  gate: string | undefined,
  a: AnalystAnalysis,
  book: number | null | undefined,
  rs: AnalystRiskSettings | undefined,
): { tone: Tone; text: string } {
  const score = Math.abs(a.score).toFixed(2)
  const name = coin(a.symbol)
  if (!gate) return { tone: 'wait', text: 'Not checked yet: the next run is within a minute.' }
  if (gate.startsWith('score<')) {
    const floor = Number(gate.slice(6))
    return {
      tone: 'wait',
      text: `Waiting: score ${score} is below ${Number.isFinite(floor) ? floor.toFixed(2) : gate.slice(6)}.`,
    }
  }
  switch (gate) {
    case 'wait': {
      const lb = a.live_bar
      if (lb && lb.held_s < lb.needed_s) {
        return {
          tone: 'wait',
          text: `Waiting: a ${lb.call} call is forming and has held ${Math.round(lb.held_s / 60)} of ${Math.round(lb.needed_s / 60)} min.`,
        }
      }
      return { tone: 'wait', text: `Waiting: no clear call yet (score ${fmtSigned(a.score)}).` }
    }
    case 'no_entry':
      return { tone: 'wait', text: 'Waiting: the call has no entry plan yet.' }
    case 'candidate':
      return { tone: 'ok', text: 'Ready: this coin is a candidate this minute.' }
    case 'kill_switch':
      return { tone: 'stop', text: 'Blocked: the kill switch is on, so no new trades open.' }
    case 'trade_open':
      return { tone: 'block', text: 'Blocked: an Analyst trade is already open (one at a time).' }
    case 'no_equity':
      return { tone: 'block', text: 'Blocked: the account balance could not be read this minute.' }
    case 'book_disagrees':
      return {
        tone: 'block',
        text: `Blocked: the Analyst says ${a.decision} but the trend book ${bookWords(book)} on ${name}. It only trades in the book's direction.`,
      }
    case 'ml_gate':
      return {
        tone: 'block',
        text: `Waiting: the AI does not rate ${name} highly enough today (it trades only on coins the AI rates 1.5x).`,
      }
    case 'book_holds_coin':
      return { tone: 'block', text: `Blocked: the trend book already holds ${name} (one position per coin).` }
    case 'too_risky_to_size':
      return {
        tone: 'block',
        text: `Blocked: one ${name} contract would risk more than ${pct(rs?.risk_cap)} of the account, or liquidation would sit too close to the stop.`,
      }
    case 'not_filled':
      return { tone: 'stop', text: 'Order not filled: the exchange rejected it or did not fill it.' }
    case 'outranked':
      return { tone: 'wait', text: 'Skipped: a stronger call on another coin was taken.' }
    case 'OPENED':
      return { tone: 'ok', text: rs?.executes ? 'Opened a trade this minute.' : 'Opened a paper trade this minute.' }
    default: {
      const w = gate.replace(/_/g, ' ')
      return { tone: 'wait', text: w.charAt(0).toUpperCase() + w.slice(1) + '.' }
    }
  }
}

export function WhyNoTrade({ data }: { data: AnalystSnapshot }) {
  const gates = data.gates
  const rs = data.risk_settings
  return (
    <Panel
      title="Why no trade?"
      right={<span className="text-[11px] text-neutral-400">latest run {data.updated ? formatISTShort(data.updated) : '--'}</span>}
      bodyClassName="px-4 py-2"
    >
      {gates === undefined ? (
        <Empty>The server is on an older version that does not report this yet.</Empty>
      ) : data.analyses.length === 0 ? (
        <Empty>No analysis yet.</Empty>
      ) : (
        <ul className="divide-y divide-neutral-800">
          {data.analyses.map((a) => {
            const g = gateText(gates[a.symbol], a, data.book_signals?.[a.symbol], rs)
            return (
              <li key={a.symbol} className="flex items-start gap-3 py-2.5">
                <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${TONE[g.tone].dot}`} aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-medium text-neutral-100">{coin(a.symbol)}</span>
                    <span className="font-mono text-[11px] text-neutral-400">
                      {a.decision} · {fmtSigned(a.score)} · {a.conviction}/10
                    </span>
                  </div>
                  <div className={`text-[13px] leading-snug ${TONE[g.tone].text}`}>{g.text}</div>
                </div>
              </li>
            )
          })}
        </ul>
      )}
      {rs && rs.executes === false && (
        <div className="border-t border-neutral-800 py-2 text-[11px] text-neutral-400">
          Engine is &ldquo;{rs.trading_engine}&rdquo;: Analyst calls open paper trades only.
        </div>
      )}
    </Panel>
  )
}

function Setting({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-neutral-800 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="font-mono text-base font-semibold tabular-nums text-neutral-100">{value}</div>
      {sub && <div className="text-[11px] leading-snug text-neutral-400">{sub}</div>}
    </div>
  )
}

const MAKER_WORDS: Record<string, string> = { all: 'ON', paper: 'Paper only', off: 'OFF' }

export function RiskSettings({ rs }: { rs: AnalystRiskSettings | undefined }) {
  if (!rs || Object.keys(rs).length === 0) return null
  const lev = rs.target_leverage ?? 0
  const trail = rs.trail_mode
    ? rs.trail_mode.replace(/_/g, ' ')
    : rs.trail_atr != null
      ? `${rs.trail_atr}× ATR`
      : '--'
  return (
    <Panel title="Risk settings in force" bodyClassName="p-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Setting
          label="Risk cap"
          value={pct(rs.risk_cap)}
          sub={`most one trade may lose at the stop; aim ${pct(rs.risk_pct)}`}
        />
        <Setting
          label="Leverage"
          value={lev > 0 ? `${lev}× target` : 'Lowest that fits'}
          sub={
            lev > 0
              ? `clamped to safe: liquidation stays ≥ ${rs.min_liq_to_stop ?? '--'}× the stop distance`
              : 'just enough margin for the contract'
          }
        />
        <Setting
          label="Maker orders"
          value={MAKER_WORDS[rs.maker_execution ?? ''] ?? rs.maker_execution ?? '--'}
          sub="limit orders on trend-book rebalances; Analyst entries are market orders"
        />
        <Setting
          label="Trail"
          value={trail}
          sub={
            rs.trail_atr != null
              ? `stop trails ${rs.trail_atr}× daily ATR, ${rs.trail_atr_tight ?? '--'}× once +1R; never moves back`
              : undefined
          }
        />
      </div>
      <div className="mt-2 text-[11px] text-neutral-400">
        Entry floor |score| ≥ {rs.min_score?.toFixed(2) ?? '--'} · engine {rs.trading_engine ?? '--'}
        {rs.live_bar ? ' · live-bar mode on' : ''}
      </div>
    </Panel>
  )
}

const MODE_BADGE: Record<string, { label: string; cls: string; words: string }> = {
  shadow: {
    label: 'SHADOW',
    cls: 'border-amber-700 bg-amber-500/15 text-amber-300',
    words: 'Logged only: these numbers have no effect on trades.',
  },
  live: {
    label: 'LIVE',
    cls: 'border-red-800 bg-red-500/15 text-red-300',
    words: 'Live: each coin’s trend-book size is multiplied by this number.',
  },
  off: { label: 'OFF', cls: 'border-neutral-700 bg-neutral-800 text-neutral-400', words: 'Off: nothing is computed.' },
}

function multCls(m: number): string {
  return m > 1 ? 'text-emerald-400' : m < 1 ? 'text-red-400' : 'text-neutral-300'
}
const multText = (m: number) => `${m % 1 === 0 ? m.toFixed(0) : m.toFixed(1)}×`

function dayLabel(r: BookFilterRecord): string {
  return new Date(r.decision_ts * 1000).toLocaleDateString('en-GB', { timeZone: 'UTC', day: '2-digit', month: 'short' })
}

function multReason(r: BookFilterRecord, sym: string): string {
  const c = r.coins[sym]
  if (!c || c.p == null) return 'Inputs were stale, so 1× (no change).'
  const lo = r.lo ?? 0
  const hi = r.hi ?? 1
  const odds = `${pct(c.p)} chance the book’s ${coin(sym)} position makes money over the next 3 days`
  const verdict =
    c.m > 1
      ? `above the ${pct(hi)} line, so size up`
      : c.m < 1
        ? `below the ${pct(lo)} line, so half size`
        : `between ${pct(lo)} and ${pct(hi)}, so normal size`
  const flat = c.S != null && Math.abs(c.S) < BOOK_LINE ? ' The book was flat in this coin, so it changes nothing.' : ''
  return `${odds}: ${verdict}.${flat}`
}

export function AiSizingPanel() {
  const q = useQuery({ queryKey: ['ml-book-filter'], queryFn: () => api.mlBookFilter(14), refetchInterval: 300_000 })
  const d = q.data
  const mode = MODE_BADGE[d?.mode ?? ''] ?? MODE_BADGE.off
  const latest = d?.latest ?? null
  const now = Date.now() / 1000
  const inForce =
    latest != null && (latest.applies_from_ts ?? 0) <= now && now < (latest.applies_until_ts ?? 0)
  const rows = [...(d?.history ?? [])].reverse().slice(0, 14)
  return (
    <Panel
      title={`AI sizing (${d ? mode.label.toLowerCase() : 'shadow'})`}
      right={d && <span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${mode.cls}`}>{mode.label}</span>}
      bodyClassName="p-4"
    >
      {q.isLoading ? (
        <Loading className="h-32" label="Loading AI sizing" />
      ) : q.isError ? (
        <ErrorState what="AI sizing" error={q.error} onRetry={() => q.refetch()} />
      ) : !d ? null : (
        <>
          <p className="text-[13px] leading-snug text-neutral-300">
            {mode.words} Once a day (00:00 UTC) a model scores each major and picks 0.5× / 1× / 1.5× for the trend
            book&rsquo;s size in that coin, for 01:00&ndash;01:00 UTC.
          </p>
          {!latest ? (
            <Empty>No daily record yet: the first one is computed just after 00:00 UTC.</Empty>
          ) : (
            <>
              <div className="mt-3 text-[11px] uppercase tracking-wide text-neutral-400">
                {dayLabel(latest)} {inForce ? '· in force now' : '· latest record'}
                {latest.stale ? ' · stale, so 1× everywhere' : ''}
              </div>
              <ul className="mt-1 divide-y divide-neutral-800">
                {MAJORS.filter((s) => latest.coins[s]).map((s) => (
                  <li key={s} className="flex items-start gap-3 py-2">
                    <span className="w-10 shrink-0 font-medium text-neutral-100">{coin(s)}</span>
                    <span className={`w-12 shrink-0 font-mono text-lg font-semibold tabular-nums ${multCls(latest.coins[s].m)}`}>
                      {multText(latest.coins[s].m)}
                    </span>
                    <span className="min-w-0 text-[12px] leading-snug text-neutral-300">{multReason(latest, s)}</span>
                  </li>
                ))}
              </ul>
              {latest.stale && (latest.reasons?.length ?? 0) > 0 && (
                <div className="mt-1 text-[11px] text-amber-300/90">Stale: {latest.reasons!.join('; ')}</div>
              )}
            </>
          )}
          {rows.length > 0 && (
            <>
              <div className="mt-4 text-[11px] uppercase tracking-wide text-neutral-400">Last {rows.length} days</div>
              <table className="mt-1 w-full text-[12px]">
                <thead>
                  <tr className="border-b border-neutral-800 text-left text-[10px] uppercase tracking-wider text-neutral-400">
                    <th className="py-1.5 pr-2 font-medium">Day (UTC)</th>
                    {MAJORS.map((s) => (
                      <th key={s} className="py-1.5 text-right font-medium">
                        {coin(s)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.decision_ts} className="border-b border-neutral-800/60">
                      <td className="py-1 pr-2 text-neutral-300">{dayLabel(r)}</td>
                      {MAJORS.map((s) => {
                        const c = r.coins[s]
                        return (
                          <td key={s} className={`py-1 text-right font-mono tabular-nums ${c ? multCls(c.m) : 'text-neutral-500'}`}>
                            {r.stale ? <span className="font-sans text-[10px] text-neutral-500">stale</span> : c ? multText(c.m) : '--'}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </Panel>
  )
}

/** Strategy race: the shadow variants' PAPER books next to the live record over the same span. */
export function StrategyRace({ race }: { race: AnalystVariantRace | undefined }) {
  if (!race?.enabled) return null
  return (
    <Panel title="Strategy race (paper)" bodyClassName="p-4">
      <p className="text-[13px] leading-snug text-neutral-300">
        A forward test of two near-miss fixes from the loss autopsy, on paper only: C skips entries when the coin&rsquo;s
        volatility (daily ATR rank) is in its top third; B sells only at score &le; &minus;0.40. Each keeps its own book,
        one trade per coin, with the live entry and exit rules. &ldquo;Base&rdquo; is the same book with no extra rule
        (the fair control). None of these send orders or count in the live stats above.
        {race.since ? ` Running since ${formatISTShort(race.since)}.` : ' No shadow trade yet.'}
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-neutral-800 text-left text-[10px] uppercase tracking-wider text-neutral-400">
              <th className="py-1.5 pr-2 font-medium">Book</th>
              <th className="py-1.5 text-right font-medium">Trades</th>
              <th className="py-1.5 text-right font-medium">Open</th>
              <th className="py-1.5 text-right font-medium">Win %</th>
              <th className="py-1.5 text-right font-medium">Avg R</th>
              <th className="py-1.5 text-right font-medium">Total R</th>
            </tr>
          </thead>
          <tbody>
            {race.rows.map((r) => (
              <tr key={r.key} className="border-b border-neutral-800/60">
                <td className={`py-1 pr-2 ${r.key === 'live' ? 'font-medium text-neutral-100' : 'text-neutral-300'}`}>{r.label}</td>
                <td className="py-1 text-right font-mono tabular-nums text-neutral-200">{r.closed}</td>
                <td className="py-1 text-right font-mono tabular-nums text-neutral-400">{r.open}</td>
                <td className="py-1 text-right font-mono tabular-nums text-neutral-200">{pct(r.win_rate)}</td>
                <td className={`py-1 text-right font-mono tabular-nums ${pnlClass(r.avg_r)}`}>{fmtR(r.avg_r)}</td>
                <td className={`py-1 text-right font-mono tabular-nums ${pnlClass(r.total_r)}`}>{fmtR(r.total_r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-[11px] text-neutral-500">
        Trades = closed trades; R net of 7 bps/side. Small samples mean nothing: judge only after many trades.
      </div>
    </Panel>
  )
}
