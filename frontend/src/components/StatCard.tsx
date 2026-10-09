export function StatCard({
  label,
  value,
  sub,
  tone = 'default',
}: {
  label: string
  value: string
  sub?: string
  tone?: 'default' | 'positive' | 'negative'
}) {
  const toneClass =
    tone === 'positive' ? 'text-emerald-400' : tone === 'negative' ? 'text-red-400' : 'text-neutral-100'
  return (
    <div className="min-w-0 rounded-xl border border-neutral-800 bg-neutral-900 p-3 sm:p-4">
      <div className="text-[11px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className={`mt-1 truncate font-mono text-lg font-semibold tabular-nums sm:text-2xl ${toneClass}`} title={value}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-neutral-400">{sub}</div>}
    </div>
  )
}
