export interface SignalTier {
  label: string
  badgeClass: string
}

// Same 0.15 entry threshold the trading loop itself uses (apply_signal_hysteresis's
// entry_threshold / SignalEnsemble's deadband) -- "Neutral" here means the same
// thing it means live: no position would actually open at this signal level.
export function signalTier(signal: number): SignalTier {
  const abs = Math.abs(signal)
  if (abs < 0.15) return { label: 'Neutral', badgeClass: 'bg-neutral-800 text-neutral-400' }
  const isLong = signal > 0
  if (abs < 0.4) {
    return isLong
      ? { label: 'Bullish', badgeClass: 'bg-emerald-950 text-emerald-400 border border-emerald-900' }
      : { label: 'Bearish', badgeClass: 'bg-red-950 text-red-400 border border-red-900' }
  }
  if (abs < 0.7) {
    return isLong
      ? { label: 'More Bullish', badgeClass: 'bg-emerald-900 text-emerald-300 border border-emerald-800' }
      : { label: 'More Bearish', badgeClass: 'bg-red-900 text-red-300 border border-red-800' }
  }
  return isLong
    ? { label: 'Ultra Bullish', badgeClass: 'bg-emerald-600 text-emerald-50' }
    : { label: 'Ultra Bearish', badgeClass: 'bg-red-600 text-red-50' }
}
