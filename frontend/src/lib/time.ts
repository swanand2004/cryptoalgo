const IST_TIME_ZONE = 'Asia/Kolkata'

/** Formats an ISO timestamp as Indian Standard Time, regardless of the
 * viewer's own browser/system timezone -- the trading system runs against
 * Indian markets, so times should read consistently as IST everywhere.
 */
export function formatIST(iso: string | null | undefined, opts: Intl.DateTimeFormatOptions = {}): string {
  if (!iso) return '--'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString('en-IN', {
    timeZone: IST_TIME_ZONE,
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
    ...opts,
  })
}

/** True if the given ISO timestamp falls on today's calendar date in IST --
 * used to scope "today" stats (e.g. Realized PnL) to the current IST day
 * regardless of the viewer's own timezone.
 */
export function isTodayIST(iso: string | null | undefined): boolean {
  if (!iso) return false
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return false
  const asIstDateKey = (d: Date) => d.toLocaleDateString('en-CA', { timeZone: IST_TIME_ZONE })
  return asIstDateKey(date) === asIstDateKey(new Date())
}

/** Converts a `YYYY-MM-DD` date (as picked in an IST-labeled date input) to
 * the UTC instant at the START of that calendar day in IST -- explicit
 * +05:30 offset, so the browser's own timezone never matters. */
export function istDateStartUtc(dateStr: string): string {
  return new Date(`${dateStr}T00:00:00+05:30`).toISOString()
}

/** Same, but the END (23:59:59.999) of that IST calendar day -- pairs with
 * istDateStartUtc for an inclusive "from this day through that day" range. */
export function istDateEndUtc(dateStr: string): string {
  return new Date(`${dateStr}T23:59:59.999+05:30`).toISOString()
}

/** Compact form for dense tables: "13 Aug, 05:41 PM IST". */
export function formatISTShort(iso: string | null | undefined): string {
  if (!iso) return '--'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '--'
  const formatted = date.toLocaleString('en-IN', {
    timeZone: IST_TIME_ZONE,
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  })
  return `${formatted} IST`
}
