const BASE = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export interface ProductInfo {
  symbol: string
  product_id: number
  underlying_asset: string
  tick_size: number
  contract_value: number
  default_leverage: number
  spot_index_symbol: string
}

export interface TickerInfo {
  symbol: string
  mark_price: number | null
  spot_price: number | null
  funding_rate: { funding_rate: number; next_funding_realization: number } | null
  last_price: number | null
  raw: Record<string, unknown>
}

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface OrderbookView {
  symbol: string
  bids: [number, number][]
  asks: [number, number][]
  best_bid: number | null
  best_ask: number | null
  mid: number | null
  imbalance: number | null
}

export interface KillSwitchReason {
  type: 'profit_target' | 'daily_loss' | 'manual' | 'max_drawdown'
  message: string
  pnl_usd: number
  pnl_pct: number
  starting_equity: number
  equity: number
}

export interface Position {
  symbol: string
  size: number
  entry_price: number
  unrealized_pnl: number
  realized_pnl: number
  strategy: string | null
  entry_time: string | null
  leverage: number | null
  fees_paid: number
  stop_loss_price: number | null
  take_profit_price: number | null
  notional_usd: number
  margin_usd: number
}

export interface OrderRow {
  id: number
  symbol: string
  side: string
  order_type: string
  size: number
  unfilled_size: number
  limit_price: number | null
  state: string
  strategy: string | null
  created_at: string
}

export interface FillRow {
  id: number
  symbol: string
  side: string
  size: number
  price: number
  fee: number
  role: string
  created_at: string
}

export interface ClosedTradeRow {
  id: number
  symbol: string
  side: string
  strategy: string | null
  size: number
  leverage: number | null
  entry_price: number
  entry_time: string
  exit_price: number
  exit_time: string
  exit_reason: string
  realized_pnl: number
  fees: number
  notional_usd: number | null
  margin_usd: number | null
}

/** realized_pnl is gross price PnL only -- fees are tracked as a separate
 * field so account balance can net them independently. Wherever "PnL" is
 * shown for a closed trade, it should be this (fee-inclusive), not the raw
 * gross field, so the number actually reflects money kept, not just price
 * movement.
 */
export function netPnl(trade: Pick<ClosedTradeRow, 'realized_pnl' | 'fees'>): number {
  return trade.realized_pnl - trade.fees
}

export interface StrategyConfigRow {
  name: string
  enabled: boolean
  weight: number
  updated_at: string
}

export interface EquityPoint {
  timestamp?: string
  time?: number
  equity: number
}

export interface MlPrediction {
  score: number | null
  decidable: boolean
  resolution: string
}

export interface GoldPrediction {
  score: number | null
  decidable: boolean
  resolution: string
  horizon_bars: number
  action: 'BUY' | 'SELL' | 'HOLD'
  target_price: number | null
  expected_move_pct: number | null
  valid_until: string
  n_samples: number | null
  mean_directional_accuracy: number | null
  mean_information_coefficient: number | null
  risk_level: 'low' | 'medium' | 'high' | 'very_high'
  risk_reason: string
}

export interface GoldSeasonality {
  month: string
  effect: 'positive' | 'negative' | 'neutral'
  note: string
  source: string
}

export interface GoldTithiEvent {
  days_away: number | null
  date: string | null
  label: string
}

export interface GoldMacroEvent {
  event: 'FOMC' | 'CPI' | 'NFP' | null
  hours_away: number | null
  at: string | null
}

export interface GoldEvents {
  seasonality: GoldSeasonality
  gold_buying_tithis: {
    shukla_tritiya: GoldTithiEvent
    krishna_trayodashi: GoldTithiEvent
  }
  next_macro_event: GoldMacroEvent
}

export interface GoldCryptoAsset {
  correlation: number | null
  interpretation: 'moving_with' | 'moving_against' | 'loosely_with' | 'loosely_against' | 'decoupled' | 'unknown'
  recent_return_pct: number | null
}

export interface GoldCrossAsset {
  ready: boolean
  resolution: string | null
  window_bars: number | null
  assets: Record<string, GoldCryptoAsset>
}

export interface GoldSnapshot {
  ready: boolean
  as_of: string | null
  price: number | null
  regime: 'trending' | 'mean_reverting' | 'random_walk' | 'unknown'
  indicators: Record<string, number>
  predictions: Record<string, GoldPrediction>
  events?: GoldEvents
  cross_asset?: GoldCrossAsset
}

export const api = {
  symbols: () => req<ProductInfo[]>('/market/symbols'),
  ticker: (symbol: string) => req<TickerInfo>(`/market/${symbol}/ticker`),
  orderbook: (symbol: string) => req<OrderbookView>(`/market/${symbol}/orderbook`),
  candles: (symbol: string, resolution = '15m', limit = 300) =>
    req<Candle[]>(`/market/${symbol}/candles?resolution=${resolution}&limit=${limit}`),

  mode: () =>
    req<{ mode: string; kill_switch: boolean; kill_switch_reason: KillSwitchReason | null }>('/trading/mode'),
  setMode: (mode: 'paper' | 'live') =>
    req<{ mode: string }>('/trading/mode', { method: 'POST', body: JSON.stringify({ mode }) }),
  killSwitch: (on: boolean) =>
    req<{ on: boolean }>('/trading/kill_switch', { method: 'POST', body: JSON.stringify({ on }) }),
  positions: () => req<Position[]>('/trading/positions'),
  equity: () => req<{ equity: number; mode: string }>('/trading/equity'),
  equityCurve: (mode: string) => req<EquityPoint[]>(`/trading/equity_curve?mode=${mode}`),
  orders: (mode: string, symbol?: string) =>
    req<OrderRow[]>(`/trading/orders?mode=${mode}${symbol ? `&symbol=${symbol}` : ''}`),
  fills: (mode: string, symbol?: string) =>
    req<FillRow[]>(`/trading/fills?mode=${mode}${symbol ? `&symbol=${symbol}` : ''}`),
  closedTrades: (mode: string, symbol?: string, dateFromUtc?: string, dateToUtc?: string) =>
    req<ClosedTradeRow[]>(
      `/trading/closed_trades?mode=${mode}${symbol ? `&symbol=${symbol}` : ''}` +
        `${dateFromUtc ? `&date_from=${encodeURIComponent(dateFromUtc)}` : ''}` +
        `${dateToUtc ? `&date_to=${encodeURIComponent(dateToUtc)}` : ''}`
    ),
  mlPredictions: () => req<Record<string, Record<string, MlPrediction>>>('/trading/ml_predictions'),

  strategies: () => req<StrategyConfigRow[]>('/strategies'),
  updateStrategy: (name: string, body: { enabled?: boolean; weight?: number }) =>
    req<StrategyConfigRow>(`/strategies/${name}`, { method: 'PUT', body: JSON.stringify(body) }),

  goldSnapshot: () => req<GoldSnapshot>('/gold/snapshot'),
  goldCandles: (resolution = '1h', limit = 300) =>
    req<Candle[]>(`/gold/candles?resolution=${resolution}&limit=${limit}`),
}
