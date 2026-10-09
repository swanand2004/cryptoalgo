// Options page API: read-only Delta chain + the PAPER S7 tracker.
// Mirrors lib/api.ts's private `req` helper (same /api base; dashboard auth
// is the browser's Basic-auth session, so no extra headers are needed).
const BASE = '/api'

async function req<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' } })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export type Underlying = 'BTC' | 'ETH'

export interface OptionQuote {
  symbol: string
  type: 'call' | 'put'
  strike: number
  expiry: string
  mark: number | null
  iv: number | null
  bid: number | null
  ask: number | null
  bid_iv: number | null
  ask_iv: number | null
  delta: number | null
  gamma: number | null
  theta: number | null
  vega: number | null
  oi_contracts: number | null
  oi: number | null
  volume: number | null
  spot: number | null
  contract_value: number | null
}

export interface OptionChain {
  underlying: Underlying
  note: string
  fetched_at: string
  spot: number | null
  expiries: { code: string; expiry: string; days: number; count: number }[]
  expiry: string | null
  expiry_time: string | null
  contract_value: number | null
  strikes: { strike: number; call: OptionQuote | null; put: OptionQuote | null }[]
}

export interface S7Row {
  id: number
  week: string
  underlying: Underlying
  signal: number | null
  signal_source: string | null
  decision: 'CALL' | 'PUT' | 'NONE'
  note: string | null
  entry_time: string | null
  symbol: string | null
  strike: number | null
  expiry: string | null
  contract_value: number | null
  spot_at_entry: number | null
  delta_at_entry: number | null
  iv_at_entry: number | null
  premium: number | null
  premium_source: string | null
  entry_fee: number | null
  cost_usd: number | null
  is_open: boolean
  settled_at: string | null
  underlying_at_expiry: number | null
  settle_value: number | null
  settle_source: string | null
  settle_fee: number | null
  pnl_usd: number | null
  pnl_pct: number | null
}

export interface S7Snapshot {
  note: string
  rules: Record<string, string | number>
  next_entry: string
  stats: {
    weeks: number
    trades: number
    open: number
    settled: number
    wins: number
    total_pnl_usd: number
    total_cost_usd: number
    return_on_premium: number | null
  }
  rows: S7Row[]
}

export const optionsApi = {
  chain: (underlying: Underlying, expiry?: string | null, width = 10) => {
    const q = new URLSearchParams({ underlying, width: String(width) })
    if (expiry) q.set('expiry', expiry)
    return req<OptionChain>(`/options/chain?${q.toString()}`)
  },
  s7: () => req<S7Snapshot>('/options/s7'),
}
