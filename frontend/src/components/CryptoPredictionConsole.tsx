import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type CryptoFactor, type CryptoPrediction } from '../lib/api'

// The blended crypto call, on the News page because news is one of its seven
// factor buckets and the reader is already here looking at drivers.
//
// The design job here is honesty under a lot of visual energy. The blend is
// 40% model / 60% factors, but only TWO of the twenty factors earned a vote,
// and the other eighteen are displayed at zero weight. A layout that renders
// all twenty as equal glowing cells would imply a breadth of evidence that
// does not exist -- so voting factors are large and lit, and the rest are a
// dim grid that says "measured, did not qualify".

const BUCKET_LABEL: Record<string, string> = {
  micro: 'Microstructure',
  macro: 'Macro',
  sentiment: 'Sentiment',
  funding: 'Funding',
  events: 'Events',
  astrology: 'Astrology',
  news: 'News',
}

const ACTION_TONE: Record<string, { text: string; bg: string; ring: string; glow: string }> = {
  BUY: {
    text: 'text-emerald-300',
    bg: 'from-emerald-500/12',
    ring: 'border-emerald-500/30',
    glow: 'rgba(52,211,153,0.35)',
  },
  SELL: {
    text: 'text-rose-300',
    bg: 'from-rose-500/12',
    ring: 'border-rose-500/30',
    glow: 'rgba(251,113,133,0.35)',
  },
  HOLD: {
    text: 'text-neutral-300',
    bg: 'from-neutral-500/8',
    ring: 'border-neutral-700',
    glow: 'rgba(148,163,184,0.22)',
  },
}

const fmtPrice = (v: number | null | undefined) =>
  v == null ? '--' : `$${v.toLocaleString(undefined, { maximumFractionDigits: v < 10 ? 4 : 2 })}`

/** Horizontal signal meter with the zero line drawn brightest.
 *
 * These scores live close to zero. Drawing the centre as the reference makes
 * "barely an opinion" read as barely an opinion, which is what most of them
 * honestly are. */
function ScoreMeter({ value, tint, height = 6 }: { value: number; tint: string; height?: number }) {
  const v = Math.max(-1, Math.min(1, value))
  const pct = Math.abs(v) * 50
  return (
    <div
      className="relative w-full overflow-hidden rounded-full bg-black/50 ring-1 ring-inset ring-white/5"
      style={{ height }}
    >
      <div className="absolute bottom-0 left-1/2 top-0 w-px bg-white/25" />
      <div
        className="absolute bottom-0 top-0 rounded-full transition-all duration-700"
        style={{
          ...(v >= 0 ? { left: '50%' } : { right: '50%' }),
          width: `${pct}%`,
          background: tint,
          boxShadow: `0 0 12px ${tint}`,
        }}
      />
    </div>
  )
}

function VotingFactor({ f }: { f: CryptoFactor }) {
  const tint = f.score > 0.05 ? '#34d399' : f.score < -0.05 ? '#fb7185' : '#94a3b8'
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.03] p-2.5" title={f.detail}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[11px] text-neutral-300">{f.name.replace(/_/g, ' ')}</span>
        <span className="font-mono text-[11px] tabular-nums" style={{ color: tint }}>
          {f.score >= 0 ? '+' : ''}
          {f.score.toFixed(2)}
        </span>
      </div>
      <div className="mt-1.5">
        <ScoreMeter value={f.score} tint={tint} height={4} />
      </div>
      <div className="mt-1 text-[9px] uppercase tracking-[0.14em]" style={{ color: tint, opacity: 0.7 }}>
        {f.direction}
      </div>
    </div>
  )
}

/** The eighteen that were measured and did not qualify. Dim, dense, honest. */
function SilentFactors({ factors }: { factors: CryptoFactor[] }) {
  const [open, setOpen] = useState(false)
  const byBucket = factors.reduce<Record<string, CryptoFactor[]>>((acc, f) => {
    ;(acc[f.bucket] ||= []).push(f)
    return acc
  }, {})

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 text-[10px] uppercase tracking-[0.14em] text-neutral-500 hover:text-neutral-300"
      >
        <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>&#9656;</span>
        {factors.length} measured &middot; zero weight
        <span className="ml-auto flex gap-1">
          {Object.keys(byBucket).map((b) => (
            <span key={b} className="rounded bg-white/5 px-1.5 py-0.5 text-[9px] normal-case tracking-normal">
              {BUCKET_LABEL[b] ?? b}
            </span>
          ))}
        </span>
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {Object.entries(byBucket).map(([bucket, fs]) => (
            <div key={bucket}>
              <div className="mb-1 text-[9px] uppercase tracking-[0.14em] text-neutral-600">
                {BUCKET_LABEL[bucket] ?? bucket}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 sm:grid-cols-3">
                {fs.map((f) => (
                  <div key={f.name} className="flex items-baseline justify-between text-[10px]" title={f.detail}>
                    <span className="truncate text-neutral-600">{f.name.replace(/_/g, ' ')}</span>
                    <span className="ml-2 shrink-0 font-mono tabular-nums text-neutral-700">
                      {f.z_score == null ? '--' : `${f.z_score >= 0 ? '+' : ''}${f.z_score.toFixed(1)}σ`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function PredictionCard({ p }: { p: CryptoPrediction }) {
  if (!p.ready) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-4">
        <div className="text-sm text-neutral-300">{p.symbol}</div>
        <div className="mt-1 text-[11px] text-neutral-600">{p.reason ?? 'not ready'}</div>
      </div>
    )
  }

  const tone = ACTION_TONE[p.action] ?? ACTION_TONE.HOLD
  const voting = p.factors.filter((f) => f.contributes)
  const silent = p.factors.filter((f) => !f.contributes)

  return (
    <div
      className={`relative overflow-hidden rounded-lg border bg-gradient-to-b ${tone.bg} to-transparent ${tone.ring}`}
      style={{ boxShadow: `inset 0 1px 0 rgba(255,255,255,0.05), 0 0 30px -18px ${tone.glow}` }}
    >
      {/* scanline sweep, purely atmospheric and motion-safe */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px overflow-hidden bg-white/5">
        <div className="pulse-sweep h-px w-1/3" style={{ background: `linear-gradient(90deg,transparent,${tone.glow},transparent)` }} />
      </div>

      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-neutral-500">{p.symbol}</div>
            <div className="mt-0.5 font-mono text-xl tabular-nums text-neutral-100">{fmtPrice(p.price)}</div>
          </div>
          <div className="text-right">
            <div className={`font-mono text-2xl font-bold tracking-[0.12em] ${tone.text}`}>{p.action}</div>
            <div className="font-mono text-[10px] tabular-nums text-neutral-500">
              {p.blended_score >= 0 ? '+' : ''}
              {p.blended_score.toFixed(3)} &middot; {p.horizon_hours}h
            </div>
          </div>
        </div>

        <div className="mt-3">
          <ScoreMeter value={p.blended_score} tint={p.blended_score >= 0 ? '#34d399' : '#fb7185'} />
        </div>

        {/* the split, as a real split */}
        <div className="mt-3 space-y-1.5">
          {[
            { label: 'Model', v: p.model_score, w: p.model_weight },
            { label: 'Factors', v: p.factor_score, w: p.factor_weight },
          ].map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span className="w-14 shrink-0 text-[10px] uppercase tracking-wider text-neutral-500">{r.label}</span>
              <span className="w-9 shrink-0 font-mono text-[10px] tabular-nums text-neutral-600">
                {Math.round(r.w * 100)}%
              </span>
              <div className="flex-1">
                <ScoreMeter value={r.v} tint={r.v >= 0 ? '#34d399aa' : '#fb7185aa'} height={4} />
              </div>
              <span className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-neutral-400">
                {r.v >= 0 ? '+' : ''}
                {r.v.toFixed(2)}
              </span>
            </div>
          ))}
        </div>

        {!p.sides_agree && (
          <div className="mt-2 inline-flex items-center gap-1.5 rounded bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-300/90">
            <span className="h-1 w-1 rounded-full bg-amber-400" />
            model and factors disagree
          </div>
        )}

        <div className="mt-3 border-t border-white/5 pt-3">
          <div className="mb-2 text-[10px] uppercase tracking-[0.14em] text-neutral-500">
            {voting.length} voting {voting.length === 1 ? 'factor' : 'factors'}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {voting.map((f) => (
              <VotingFactor key={f.name} f={f} />
            ))}
          </div>
          <SilentFactors factors={silent} />
        </div>
      </div>
    </div>
  )
}

export default function CryptoPredictionConsole() {
  const q = useQuery({
    queryKey: ['crypto_predictions'],
    queryFn: api.cryptoPredictions,
    refetchInterval: 60_000,
  })

  const data = q.data

  return (
    <div className="space-y-3">
      <style>{`
        @keyframes pulse-sweep-kf { 0% { transform: translateX(-120%); } 100% { transform: translateX(420%); } }
        .pulse-sweep { animation: pulse-sweep-kf 4.5s cubic-bezier(.4,0,.6,1) infinite; }
        @media (prefers-reduced-motion: reduce) { .pulse-sweep { animation: none; } }
      `}</style>

      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-cyan-500" />
            </span>
            <h2 className="text-sm font-medium tracking-wide text-neutral-100">Blended forecast</h2>
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-neutral-600">
              {data ? `${Math.round(data.model_weight * 100)}/${Math.round((1 - data.model_weight) * 100)} model / factors` : '--'}
            </span>
          </div>
          <p className="mt-1 max-w-2xl text-[11px] leading-relaxed text-neutral-600">
            Model plus micro, macro, sentiment, funding, events, astrology and news. Twenty factors are computed;{' '}
            <span className="text-neutral-400">two earned a vote</span> by holding up across BTC, ETH and SOL on
            held-out data. The rest are shown at zero weight rather than hidden, because a factor that failed is
            information too.
          </p>
        </div>
      </div>

      {q.isLoading && (
        <div className="rounded-lg border border-neutral-800 bg-neutral-950/60 py-10 text-center text-sm text-neutral-600">
          Computing blended forecast…
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            {data.predictions.map((p) => (
              <PredictionCard key={p.symbol} p={p} />
            ))}
          </div>
          <p className="text-[10px] leading-relaxed text-neutral-600">
            Measured on the deep Binance history (79,170 hourly BTC/ETH bars from 2017, 53,116 SOL bars from 2020),
            four hours ahead, with signs fitted on train, factors selected on validation and test read once. The two
            voting factors beat a long-only baseline by{' '}
            <span className="font-mono text-neutral-400">+1.43 / +1.58 / +1.57</span> points on BTC / ETH / SOL, all
            outside the noise band. The model half scores <span className="font-mono text-neutral-400">below 50%</span>{' '}
            out-of-fold on all three, so its 40% weight costs about 0.15 points &mdash; a design choice, kept visible
            rather than buried. News has four days of history and cannot be validated yet.
          </p>
        </>
      )}
    </div>
  )
}
