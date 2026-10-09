// One set of number formatters for every page, so a price, a dollar amount,
// a percent and an R-multiple always read the same way.

/** A market price. Small coins trade below $1, so they need more decimals. */
export function fmtPrice(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '--'
  const abs = Math.abs(v)
  if (abs >= 1000) return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  if (abs >= 100) return v.toFixed(2)
  if (abs >= 1) return v.toFixed(3)
  return v.toFixed(5)
}

/** A dollar amount: "$14.21", or "+$0.42" / "-$0.42" with `signed`. */
export function fmtUsd(v: number | null | undefined, opts: { signed?: boolean; digits?: number } = {}): string {
  if (v == null || !Number.isFinite(v)) return '--'
  const digits = opts.digits ?? 2
  const body = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
  const sign = v < 0 ? '-' : opts.signed ? '+' : ''
  return `${sign}$${body}`
}

/** A percent that is already in percent units (2.5 -> "+2.50%"). */
export function fmtPct(v: number | null | undefined, digits = 2, signed = true): string {
  if (v == null || !Number.isFinite(v)) return '--'
  return `${signed && v >= 0 ? '+' : ''}${v.toFixed(digits)}%`
}

/** An R-multiple: "+1.20R". */
export function fmtR(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '--'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}R`
}

/** A signed plain number: "+0.34". */
export function fmtSigned(v: number, digits = 2): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`
}

/** Green for gains, red for losses, grey for exactly zero / unknown. */
export function pnlClass(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v) || v === 0) return 'text-neutral-300'
  return v > 0 ? 'text-emerald-400' : 'text-red-400'
}

/** "BTCUSD" -> "BTC". */
export function coin(symbol: string): string {
  return symbol.replace(/USDT?$/, '')
}

/** How long ago an ISO time was, compactly: "3m ago", "5h ago", "2d ago". */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '--'
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return '--'
  const mins = Math.max(0, Math.round((now - t) / 60_000))
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.round(mins / 60)
  if (hours < 48) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

// Why a bot trade closed. Anything not listed is shown in plain words rather
// than lumped in with "signal".
const EXIT_REASON: Record<string, { label: string; cls: string }> = {
  stop_loss: { label: 'Stop loss', cls: 'bg-red-950 text-red-300 border-red-900' },
  take_profit: { label: 'Take profit', cls: 'bg-emerald-950 text-emerald-300 border-emerald-900' },
  signal: { label: 'Signal', cls: 'bg-neutral-800 text-neutral-300 border-neutral-700' },
  manual: { label: 'Manual', cls: 'bg-neutral-800 text-neutral-300 border-neutral-700' },
  kill_switch: { label: 'Kill switch', cls: 'bg-amber-950 text-amber-300 border-amber-900' },
  daily_loss: { label: 'Daily loss limit', cls: 'bg-red-950 text-red-300 border-red-900' },
  profit_target: { label: 'Daily profit target', cls: 'bg-emerald-950 text-emerald-300 border-emerald-900' },
  max_drawdown: { label: 'Max drawdown', cls: 'bg-red-950 text-red-300 border-red-900' },
}

export function exitReason(reason: string): { label: string; cls: string } {
  const known = EXIT_REASON[reason]
  if (known) return known
  const words = reason.replace(/_/g, ' ')
  return { label: words.charAt(0).toUpperCase() + words.slice(1), cls: EXIT_REASON.signal.cls }
}
