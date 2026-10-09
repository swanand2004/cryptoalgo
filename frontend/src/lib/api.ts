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
  signal?: number | null
  exit_threshold?: number | null
  notional_usd: number
  margin_usd: number
  // Prices at which the account limits would close this trade (no exchange stop).
  liquidation_price?: number | null
  loss_exit_price?: number | null
  profit_exit_price?: number | null
  drawdown_exit_price?: number | null
}

// ---- Gold Signal Board (technicals + Gann, direction-neutral) ----
export interface TrackRecord {
  hit: number | null
  base: number | null
  n: number
}
export interface BoardIndicator extends TrackRecord {
  key: string
  name: string
  family: 'Trend' | 'Oscillator' | 'Gann'
  detail: string
  vote: number
  since: string
  price_at_since: number
  bars_since: number
}
export interface GannLevels {
  swing: {
    last_top: { price: number; time: string; bars_ago: number } | null
    last_bottom: { price: number; time: string; bars_ago: number } | null
  }
  angles: { name: string; price: number; price_is: 'above' | 'below' }[]
  square_of_nine: { price: number; angle: number; kind: 'support' | 'resistance' }[]
  time_cycles: { due: string; bars: number; from: 'top' | 'bottom'; pivot_time: string; pivot_price: number }[]
}
export interface GoldTrade {
  side: 'BUY' | 'SELL'
  entry_time: string
  entry: number
  stop_loss: number
  take_profit: number
  risk_reward: number
  status: 'ACTIVE' | 'TP HIT' | 'STOPPED OUT'
  exit_time: string | null
  exit_price: number | null
  closed_time: string | null
  result: 'TP' | 'SL' | null
  open: boolean
  pnl_pct: number
  r_multiple: number
  bars_held: number
}
export interface SignalBoard {
  timeframe: 'daily' | 'hourly'
  horizon_bars: number
  bar_close_time: string
  price: number
  consensus: TrackRecord & {
    direction: 'BUY' | 'SELL' | 'NEUTRAL'
    score: number
    buy: number
    sell: number
    neutral: number
    since: string
    price_at_signal: number
    change_since_pct: number
    bars_since: number
  }
  indicators: BoardIndicator[]
  gann: GannLevels
  change_24h_pct: number | null
  trade: GoldTrade | null
  // Signal active but no open trade: the levels of a trade entered now.
  plan: { side: 'BUY' | 'SELL'; entry: number; stop_loss: number; take_profit: number } | null
  last_trade: GoldTrade | null
  history: GoldTrade[]
  history_stats: {
    trades: number
    since?: string
    tp?: number
    sl?: number
    win_rate?: number
    net_pct?: number
    avg_pct?: number
  }
  triggers: { buy_above: number; sell_below: number } | null
  levels: { price: number; label: string; source: 'pivot' | 'gann' | 'swing' | 'range'; kind: 'support' | 'resistance' }[]
  families: { family: string; label: string; score: number; buy: number; sell: number; neutral: number }[]
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

export interface GoldLevels {
  support: number | null
  resistance: number | null
  breakout_above: number | null
  breakdown_below: number | null
  invalidation: number | null
  atr: number | null
  range_window_bars: number | null
  distance_to_breakout_pct: number | null
  distance_to_breakdown_pct: number | null
}

export interface GoldTradePlan {
  action: 'BUY' | 'SELL' | 'HOLD'
  viable: boolean
  current_price: number | null
  entry: number | null
  entry_type?: 'market' | 'limit'
  limit_entry?: number | null
  entry_distance_pct?: number | null
  entry_distance_atr?: number | null
  entry_reachable?: boolean
  stop_loss: number | null
  take_profit: number | null
  risk_per_unit?: number | null
  reward_per_unit?: number | null
  risk_reward: number | null
  risk_reward_at_market?: number | null
  target_rr?: number
  min_viable_rr?: number
  target_remaining_usd?: number | null
  target_remaining_pct?: number | null
  stop_distance_pct?: number | null
  stop_atr_multiple?: number | null
  atr?: number | null
  structural_target?: number | null
  structural_risk_reward?: number | null
  size_multiplier?: number | null
  model_conflict?: boolean
  trigger_long_above?: number | null
  trigger_short_below?: number | null
  note: string
}

export interface GoldMacroFactor {
  name: string
  raw_value: number | null
  z_score: number | null
  score: number
  direction: 'bullish' | 'bearish' | 'neutral'
  as_of: string | null
  detail: string
  contributes: boolean
}

export interface GoldMicroFactor {
  name: string
  kind: 'directional' | 'regime'
  value: number | null
  score: number
  conviction: number
  direction: string
  detail: string
}

export interface GoldBlend {
  model_score: number
  factor_score: number
  directional_score: number
  micro_conviction: number
  micro_factors: GoldMicroFactor[]
  blended_score: number
  model_weight: number
  factor_weight: number
  factors_available: number
  factors_contributing: number
  factors_total: number
  sides_agree: boolean
  factors: GoldMacroFactor[]
}

export interface GoldPrediction {
  levels?: GoldLevels
  trade?: GoldTradePlan | null
  blend?: GoldBlend | null
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
  long_only_baseline: number | null
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

export interface CryptoFactor {
  name: string
  bucket: 'micro' | 'macro' | 'sentiment' | 'funding' | 'events' | 'astrology' | 'news'
  value: number | null
  z_score: number | null
  score: number
  direction: 'bullish' | 'bearish' | 'neutral'
  /** Did this factor survive measurement? Only these carry weight. */
  contributes: boolean
  detail: string
}

export interface CryptoPrediction {
  symbol: string
  ready: boolean
  reason?: string
  as_of: string
  price: number | null
  horizon_hours: number
  action: 'BUY' | 'SELL' | 'HOLD'
  model_score: number
  factor_score: number
  blended_score: number
  model_weight: number
  factor_weight: number
  deadband: number
  sides_agree: boolean
  model_ready?: boolean
  factors_total: number
  factors_contributing: number
  buckets: string[]
  factors: CryptoFactor[]
}

export interface CryptoPredictionsResponse {
  as_of: string
  model_weight: number
  default_model_weight: number
  resolution: string
  predictions: CryptoPrediction[]
}

export interface NewsHeadline {
  title: string
  source: string
  time: number
  sentiment: number
  label: 'bullish' | 'bearish' | 'neutral'
  symbols: string[]
}

export interface NewsHeadlinesResponse {
  fetched_at: string | null
  headlines: NewsHeadline[]
}

export interface NewsSentimentPoint {
  time: number
  sentiment: number
  headline_count: number
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
  goldSignalBoard: () => req<{ daily: SignalBoard | null; hourly: SignalBoard | null }>('/gold/signal_board'),
  goldCandles: (resolution = '1h', limit = 300) =>
    req<Candle[]>(`/gold/candles?resolution=${resolution}&limit=${limit}`),

  cryptoPredictions: () => req<CryptoPredictionsResponse>('/predictions/crypto'),
  analyst: () => req<AnalystSnapshot>('/analyst'),
  analystGold: () => req<AnalystGoldSnapshot>('/analyst/gold'),
  mlBookFilter: (days = 14) => req<BookFilterResponse>(`/ml/book-filter?days=${days}`),

  newsHeadlines: (limit = 40) => req<NewsHeadlinesResponse>(`/news/headlines?limit=${limit}`),
  newsSentimentHistory: (symbol = 'GLOBAL', hours = 72) =>
    req<NewsSentimentPoint[]>(`/news/sentiment_history?symbol=${symbol}&hours=${hours}`),
}

// ---- Analyst (free rules-based agent; paper trades only) ----
export interface AnalystFactor {
  key: string
  family: 'technical' | 'structure' | 'derivatives' | 'sentiment' | 'macro' | 'live'
  name: string
  score: number
  reading: string
}
export interface AnalystAnalysis {
  symbol: string
  time: string
  price: number
  decision: 'BUY' | 'SELL' | 'WAIT'
  score: number
  conviction: number
  size_factor: number
  families: Partial<Record<AnalystFactor['family'], number>>
  factors: AnalystFactor[]
  notes: string[]
  entry: number | null
  stop_loss: number | null
  take_profit: number | null
  atr?: number | null
  supports?: number[]
  resistances?: number[]
  plan_note?: string
  /** Live-bar mode: the call that is forming and how long it has held. */
  live_bar?: { call: 'BUY' | 'SELL'; held_s: number; needed_s: number }
}
export interface AnalystTrade {
  id: number
  symbol: string
  side: 'BUY' | 'SELL'
  conviction: number
  entry_time: string
  entry: number
  stop_loss: number
  take_profit: number
  open: boolean
  exit_time: string | null
  exit_price: number | null
  result: 'TP' | 'SL' | 'TRAIL' | 'FLIP' | 'SYNC' | null
  r_multiple: number | null
  pnl_pct: number
  live?: boolean
  contracts?: number
  extended?: boolean
}
export interface AnalystSnapshot {
  updated: string | null
  best: string | null
  analyses: AnalystAnalysis[]
  open_trades: AnalystTrade[]
  closed_trades: AnalystTrade[]
  stats: { closed: number; wins: number; win_rate: number | null; avg_r: number | null; total_r: number }
  /** Latest run's gate per coin ("wait", "score<0.2", "book_disagrees", "OPENED", ...). */
  gates?: Record<string, string>
  /** The trend book's live signal per coin (-1..+1; |s| >= 0.15 = holding). */
  book_signals?: Record<string, number | null>
  risk_settings?: AnalystRiskSettings
  /** Shadow variants (paper only) vs the live record over the same span; never in `stats`. */
  variants?: AnalystVariantRace
}
export interface AnalystVariantRow {
  key: string
  label: string
  closed: number
  open: number
  wins: number
  win_rate: number | null
  avg_r: number | null
  total_r: number
}
export interface AnalystVariantRace {
  enabled: boolean
  since: string | null
  rows: AnalystVariantRow[]
}
export interface AnalystRiskSettings {
  trading_engine?: string
  executes?: boolean
  risk_pct?: number
  risk_cap?: number
  target_leverage?: number
  min_liq_to_stop?: number
  max_leverage?: number
  maker_execution?: string
  trail_mode?: string | null
  trail_atr?: number
  trail_atr_tight?: number
  min_score?: number
  min_conviction?: number
  live_bar?: boolean
}

// ---- ML book filter v1 (daily size multiplier on the trend book; "shadow" = logged only) ----
export interface BookFilterCoin {
  p: number | null
  m: number
  S: number | null
}
export interface BookFilterRecord {
  model?: string
  decision_ts: number
  decision_time?: string
  applies_from_ts?: number
  applies_until_ts?: number
  lo?: number
  hi?: number
  stale?: boolean
  reasons?: string[]
  coins: Record<string, BookFilterCoin>
  mode?: string
  computed_at?: string
}
export interface BookFilterResponse {
  mode: string
  latest: BookFilterRecord | null
  in_force: Record<string, number>
  history: BookFilterRecord[]
  note: string
}
export interface AnalystGoldSnapshot {
  updated: string | null
  analysis: AnalystAnalysis | null
  open_trades: AnalystTrade[]
  closed_trades: AnalystTrade[]
  stats: AnalystSnapshot['stats']
}
