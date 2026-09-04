import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type NewsHeadline, type NewsSentimentPoint } from '../lib/api'
import CryptoPredictionConsole from '../components/CryptoPredictionConsole'
import { NewsGlobe } from '../components/NewsGlobe'

const TABS = [
  { key: 'ALL', label: 'All' },
  { key: 'BTCUSD', label: 'BTC' },
  { key: 'ETHUSD', label: 'ETH' },
  { key: 'SOLUSD', label: 'SOL' },
] as const

type TabKey = (typeof TABS)[number]['key']

const LABEL_STYLE: Record<string, { text: string; dot: string; chip: string }> = {
  bullish: { text: 'Bullish', dot: 'bg-emerald-400', chip: 'bg-emerald-950 text-emerald-400 border border-emerald-900' },
  bearish: { text: 'Bearish', dot: 'bg-red-400', chip: 'bg-red-950 text-red-400 border border-red-900' },
  neutral: { text: 'Neutral', dot: 'bg-cyan-400', chip: 'bg-cyan-950 text-cyan-400 border border-cyan-900' },
}

function labelFor(sentiment: number): string {
  if (sentiment >= 0.05) return 'bullish'
  if (sentiment <= -0.05) return 'bearish'
  return 'neutral'
}

function timeAgo(unixSeconds: number): string {
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - unixSeconds))
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function symbolTag(symbol: string): string {
  return symbol.replace('USD', '')
}

function SentimentSparkline({ points }: { points: NewsSentimentPoint[] }) {
  if (points.length === 0) {
    return <div className="h-10 flex items-center text-[11px] text-neutral-700">no history yet</div>
  }
  const maxAbs = Math.max(0.15, ...points.map((p) => Math.abs(p.sentiment)))
  return (
    <div className="flex items-end gap-[2px] h-10">
      {points.map((p, i) => {
        const heightPct = p.sentiment === 0 ? 3 : Math.max(8, (Math.abs(p.sentiment) / maxAbs) * 100)
        const positive = p.sentiment >= 0
        return (
          <div
            key={i}
            className="flex-1 h-full flex flex-col justify-end"
            title={`${new Date(p.time * 1000).toLocaleString()}: ${p.sentiment.toFixed(2)} (${p.headline_count} headline${p.headline_count === 1 ? '' : 's'})`}
          >
            <div className={`rounded-sm ${positive ? 'bg-emerald-500/80' : 'bg-red-500/80'}`} style={{ height: `${heightPct}%` }} />
          </div>
        )
      })}
    </div>
  )
}

function averageRecent(points: NewsSentimentPoint[], hours: number): { mean: number | null; count: number } {
  const cutoff = Date.now() / 1000 - hours * 3600
  const recent = points.filter((p) => p.time >= cutoff)
  const totalHeadlines = recent.reduce((sum, p) => sum + p.headline_count, 0)
  if (totalHeadlines === 0) return { mean: null, count: 0 }
  const weighted = recent.reduce((sum, p) => sum + p.sentiment * p.headline_count, 0)
  return { mean: weighted / totalHeadlines, count: totalHeadlines }
}

function SentimentGauge({ title, symbol }: { title: string; symbol: string }) {
  const query = useQuery({
    queryKey: ['news_sentiment_history', symbol],
    queryFn: () => api.newsSentimentHistory(symbol, 72),
    refetchInterval: 60_000,
  })
  const points = query.data ?? []
  const { mean, count } = averageRecent(points, 24)
  const label = mean == null ? null : LABEL_STYLE[labelFor(mean)]

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-neutral-400 uppercase tracking-wide">{title}</div>
        {label && <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${label.chip}`}>{label.text}</span>}
      </div>
      <div className="font-mono text-2xl tabular-nums text-neutral-100">
        {mean == null ? '--' : `${mean >= 0 ? '+' : ''}${mean.toFixed(3)}`}
      </div>
      <div className="text-[11px] text-neutral-600">{count > 0 ? `${count} headlines, last 24h` : 'no headlines in the last 24h'}</div>
      <SentimentSparkline points={points.slice(-42)} />
    </div>
  )
}

function HeadlineRow({ headline }: { headline: NewsHeadline }) {
  const style = LABEL_STYLE[headline.label] ?? LABEL_STYLE.neutral
  return (
    <div className="flex items-start gap-3 py-3 px-4 border-b border-neutral-800/70 last:border-b-0 hover:bg-neutral-800/20 transition-colors">
      <span className={`mt-1.5 h-2 w-2 rounded-full shrink-0 shadow-[0_0_6px_currentColor] ${style.dot}`} />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-neutral-200 leading-snug">{headline.title}</div>
        <div className="mt-1.5 flex items-center flex-wrap gap-2 text-[11px] text-neutral-600 font-mono">
          <span className="uppercase tracking-wide">{headline.source}</span>
          <span>&middot;</span>
          <span>{timeAgo(headline.time)}</span>
          {headline.symbols.map((s) => (
            <span key={s} className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400">
              {symbolTag(s)}
            </span>
          ))}
        </div>
      </div>
      <div className="shrink-0 text-right">
        <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${style.chip}`}>{style.text}</span>
        <div className="mt-1 font-mono text-[11px] text-neutral-600 tabular-nums">
          {headline.sentiment >= 0 ? '+' : ''}
          {headline.sentiment.toFixed(2)}
        </div>
      </div>
    </div>
  )
}

function GlobalPulsePanel({ headlines }: { headlines: NewsHeadline[] }) {
  const counts = useMemo(() => {
    const c = { bullish: 0, bearish: 0, neutral: 0 }
    for (const h of headlines) c[h.label as keyof typeof c] = (c[h.label as keyof typeof c] ?? 0) + 1
    return c
  }, [headlines])
  const total = counts.bullish + counts.bearish + counts.neutral

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-cyan-900/40 bg-neutral-950/60 p-4 shadow-[0_0_25px_-8px_rgba(34,211,238,0.25)]">
      <div>
        <div className="text-[11px] font-medium text-cyan-500 uppercase tracking-[0.2em]">Global Pulse</div>
        <div className="text-[11px] text-neutral-600 mt-1">Last {total} headlines, live feed</div>
      </div>

      <div className="space-y-2.5">
        {(['bullish', 'neutral', 'bearish'] as const).map((k) => {
          const style = LABEL_STYLE[k]
          const pct = total > 0 ? (counts[k] / total) * 100 : 0
          return (
            <div key={k}>
              <div className="flex items-center justify-between text-[11px] mb-1">
                <span className="text-neutral-400">{style.text}</span>
                <span className="font-mono text-neutral-500">{counts[k]}</span>
              </div>
              <div className="h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                <div
                  className={`h-full rounded-full ${k === 'bullish' ? 'bg-emerald-500' : k === 'bearish' ? 'bg-red-500' : 'bg-cyan-500'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="pt-3 border-t border-neutral-800/70 text-[11px] text-neutral-600 leading-relaxed">
        Hub positions are illustrative -- headlines aren't geotagged at the source, so each is placed at the
        financial center its own text most plausibly concerns, not a detected origin.
      </div>
    </div>
  )
}

export default function News() {
  const [tab, setTab] = useState<TabKey>('ALL')

  const headlinesQuery = useQuery({
    queryKey: ['news_headlines'],
    queryFn: () => api.newsHeadlines(60),
    refetchInterval: 60_000,
  })

  const headlines = headlinesQuery.data?.headlines ?? []
  const filtered = useMemo(
    () => (tab === 'ALL' ? headlines : headlines.filter((h) => h.symbols.includes(tab))),
    [headlines, tab]
  )

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="rounded-lg border border-neutral-800 bg-gradient-to-b from-neutral-900 to-neutral-900/60 p-4">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-500" />
          </span>
          <div className="text-sm font-medium text-neutral-200 uppercase tracking-wide">News &amp; Sentiment</div>
          <span className="text-[10px] text-neutral-600 uppercase tracking-wider">Global Feed</span>
        </div>
        <div className="text-xs text-neutral-600 mt-1 max-w-2xl">
          Free crypto trade press (CoinDesk + CoinTelegraph), scored with VADER sentiment. Context for reading the
          market, not a trading input yet -- there's no free historical news archive, so this only has real history
          from 2026-09-02 onward, and it isn't fed to any strategy until there's enough of it to walk-forward test
          honestly.
        </div>
      </div>

      <CryptoPredictionConsole />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-lg border border-cyan-900/40 bg-[radial-gradient(circle_at_30%_20%,rgba(8,145,178,0.08),transparent_60%)] p-2 h-[420px]">
          <NewsGlobe headlines={headlines} />
        </div>
        <GlobalPulsePanel headlines={headlines} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <SentimentGauge title="Global" symbol="GLOBAL" />
        <SentimentGauge title="Bitcoin" symbol="BTCUSD" />
        <SentimentGauge title="Ethereum" symbol="ETHUSD" />
        <SentimentGauge title="Solana" symbol="SOLUSD" />
      </div>

      <div className="rounded-lg border border-neutral-800 bg-neutral-900">
        <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
          <div className="flex gap-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  tab === t.key ? 'bg-neutral-800 text-neutral-50' : 'text-neutral-500 hover:text-neutral-300'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          {headlinesQuery.data?.fetched_at && (
            <div className="text-[11px] text-neutral-600 font-mono">
              updated {timeAgo(Math.floor(new Date(headlinesQuery.data.fetched_at).getTime() / 1000))}
            </div>
          )}
        </div>

        {headlinesQuery.isLoading && <div className="p-6 text-center text-sm text-neutral-600">Loading headlines...</div>}
        {!headlinesQuery.isLoading && filtered.length === 0 && (
          <div className="p-6 text-center text-sm text-neutral-600">No headlines for this filter right now.</div>
        )}
        {filtered.map((h, i) => (
          <HeadlineRow key={`${h.time}-${i}`} headline={h} />
        ))}
      </div>
    </div>
  )
}
