import { useEffect, useMemo, useRef, useState } from 'react'
import Globe, { type GlobeMethods } from 'react-globe.gl'
import * as THREE from 'three'
import type { NewsHeadline } from '../lib/api'

/** Home base for the "news flowing in" arcs -- this platform trades on
 * Delta Exchange India, so Mumbai is the natural receiving point, not a
 * literal claim about where the server runs. */
const HOME = { lat: 19.076, lng: 72.877, name: 'MUMBAI' }

/** Illustrative hub assignment, stated plainly: CoinDesk/CoinTelegraph
 * headlines are NOT geotagged at the source. Each headline is mapped to the
 * financial-hub its own text most plausibly concerns (via keyword match),
 * falling back to a deterministic rotation across major crypto/finance
 * centers so the globe still reads as "global" rather than piling every
 * unmatched headline on one point. This is a display device for "where in
 * the world is this news about", not a claim of detected origin.
 */
const HUBS: { name: string; lat: number; lng: number; keywords: string[] }[] = [
  { name: 'NEW YORK', lat: 40.713, lng: -74.006, keywords: ['sec', 'fed', 'washington', 'dollar', 'u.s.', 'united states', 'wall street', 'cftc', 'nasdaq'] },
  { name: 'LONDON', lat: 51.507, lng: -0.128, keywords: ['uk', 'britain', 'london', 'europe', 'eu ', 'ecb', 'euro', 'england'] },
  { name: 'SINGAPORE', lat: 1.352, lng: 103.82, keywords: ['singapore', 'mas '] },
  { name: 'TOKYO', lat: 35.676, lng: 139.65, keywords: ['japan', 'tokyo', 'yen', 'boj'] },
  { name: 'SEOUL', lat: 37.566, lng: 126.978, keywords: ['korea', 'seoul', 'won'] },
  { name: 'SHANGHAI', lat: 31.23, lng: 121.474, keywords: ['china', 'beijing', 'shanghai', 'yuan', 'pboc'] },
  { name: 'DUBAI', lat: 25.204, lng: 55.271, keywords: ['dubai', 'uae', 'vara'] },
  { name: 'MUMBAI', lat: 19.076, lng: 72.877, keywords: ['india', 'rbi', 'sebi', 'rupee', 'mumbai'] },
]
const FALLBACK_HUBS = [HUBS[0], HUBS[1], HUBS[2]] // NY / London / Singapore -- the 3 biggest crypto trading sessions

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

function hubForHeadline(headline: NewsHeadline) {
  const lowered = headline.title.toLowerCase()
  const matched = HUBS.find((h) => h.keywords.some((k) => lowered.includes(k)))
  if (matched) return matched
  return FALLBACK_HUBS[hashString(headline.title) % FALLBACK_HUBS.length]
}

function sentimentColor(sentiment: number): string {
  if (sentiment >= 0.05) return '#34d399' // emerald-400
  if (sentiment <= -0.05) return '#f87171' // red-400
  return '#67e8f9' // cyan-300, neutral reads as "signal", not "nothing"
}

/** react-globe.gl needs explicit pixel dimensions -- this just tracks the
 * wrapper div's size so the globe fills its card responsively. */
function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return { ref, size }
}

/** A fixed, seeded starfield -- generated once, not re-randomized on every
 * re-render (which would make stars visibly jitter each data refresh). */
function useStarfield(count: number): { top: string; left: string; size: number; opacity: number }[] {
  return useMemo(() => {
    let seed = 42
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }
    return Array.from({ length: count }, () => ({
      top: `${(rand() * 100).toFixed(2)}%`,
      left: `${(rand() * 100).toFixed(2)}%`,
      size: rand() > 0.85 ? 2 : 1,
      opacity: 0.25 + rand() * 0.55,
    }))
  }, [count])
}

function CornerBrackets() {
  const base = 'absolute h-4 w-4 border-cyan-500/50'
  return (
    <>
      <div className={`${base} top-2 left-2 border-l-2 border-t-2`} />
      <div className={`${base} top-2 right-2 border-r-2 border-t-2`} />
      <div className={`${base} bottom-2 left-2 border-l-2 border-b-2`} />
      <div className={`${base} bottom-2 right-2 border-r-2 border-b-2`} />
    </>
  )
}

export function NewsGlobe({ headlines }: { headlines: NewsHeadline[] }) {
  const { ref: containerRef, size } = useElementSize<HTMLDivElement>()
  const globeRef = useRef<GlobeMethods | undefined>(undefined)
  const stars = useStarfield(90)

  const globeMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#0a1a2e'),
        wireframe: true,
        transparent: true,
        opacity: 0.35,
      }),
    []
  )

  const { arcs, points } = useMemo(() => {
    const recent = headlines.slice(0, 25)
    const hubCounts = new Map<string, { hub: (typeof HUBS)[number]; count: number }>()

    const arcsOut = recent.map((h) => {
      const hub = hubForHeadline(h)
      const existing = hubCounts.get(hub.name)
      hubCounts.set(hub.name, { hub, count: (existing?.count ?? 0) + 1 })
      return {
        startLat: hub.lat,
        startLng: hub.lng,
        endLat: HOME.lat,
        endLng: HOME.lng,
        color: sentimentColor(h.sentiment),
        title: h.title,
        hubName: hub.name,
      }
    })

    const pointsOut = Array.from(hubCounts.values()).map(({ hub, count }, i) => ({
      lat: hub.lat,
      lng: hub.lng,
      name: hub.name,
      count,
      size: Math.min(1.4, 0.4 + count * 0.18),
      isHome: false,
      phase: i,
    }))
    pointsOut.push({ lat: HOME.lat, lng: HOME.lng, name: `${HOME.name} (HQ)`, count: recent.length, size: 1.1, isHome: true, phase: 0 })

    return { arcs: arcsOut, points: pointsOut }
  }, [headlines])

  useEffect(() => {
    if (size.width === 0) return
    const controls = globeRef.current?.controls()
    if (controls) {
      controls.autoRotate = true
      controls.autoRotateSpeed = 0.55
      controls.enableZoom = false
    }
  }, [size.width])

  const handleGlobeReady = () => {
    // Opens on a view that actually shows Mumbai + the Asia/Middle-East hubs
    // together, rather than react-globe.gl's default mid-Atlantic center.
    globeRef.current?.pointOfView({ lat: 18, lng: 55, altitude: 2.3 }, 0)
  }

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[380px] overflow-hidden rounded-lg bg-[#03060d]">
      {stars.map((s, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-cyan-100 pointer-events-none"
          style={{ top: s.top, left: s.left, width: s.size, height: s.size, opacity: s.opacity }}
        />
      ))}
      <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(circle_at_center,transparent_55%,rgba(3,6,13,0.75)_100%)]" />
      <CornerBrackets />
      <div className="pointer-events-none absolute top-3 left-8 z-20 flex items-center gap-1.5 font-mono text-[10px] text-cyan-500/80 uppercase tracking-widest">
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-cyan-400" />
        </span>
        Tracking {points.length} hubs
      </div>
      {size.width > 0 && (
        <Globe
          ref={globeRef}
          width={size.width}
          height={size.height}
          onGlobeReady={handleGlobeReady}
          backgroundColor="rgba(0,0,0,0)"
          globeMaterial={globeMaterial}
          showGraticules
          showAtmosphere
          atmosphereColor="#22d3ee"
          atmosphereAltitude={0.24}
          arcsData={arcs}
          arcColor={(d: any) => d.color}
          arcAltitudeAutoScale={0.32}
          arcStroke={0.4}
          arcDashLength={0.4}
          arcDashGap={2.2}
          arcDashAnimateTime={2200}
          arcLabel={(d: any) => `<div style="font-family:monospace;font-size:11px;background:#04070fcc;border:1px solid #164e63;padding:4px 8px;border-radius:4px;color:#e2e8f0;max-width:220px">${d.hubName} &rarr; MUMBAI<br/>${d.title}</div>`}
          ringsData={points}
          ringLat={(d: any) => d.lat}
          ringLng={(d: any) => d.lng}
          ringColor={(d: any) => (t: number) => `rgba(103, 232, 249, ${(d.isHome ? 0.55 : 0.35) * (1 - t)})`}
          ringMaxRadius={(d: any) => (d.isHome ? 5 : 2.6 + d.count * 0.4)}
          ringPropagationSpeed={(d: any) => (d.isHome ? 2.2 : 1.6)}
          ringRepeatPeriod={(d: any) => 1400 + (d.phase % 4) * 350}
          labelsData={points}
          labelLat={(d: any) => d.lat}
          labelLng={(d: any) => d.lng}
          labelText={(d: any) => d.name}
          labelSize={(d: any) => 0.38 + Math.min(0.3, d.count * 0.05)}
          labelDotRadius={(d: any) => 0.35 + Math.min(0.6, d.count * 0.12)}
          labelColor={(d: any) => (d.isHome ? '#a5f3fc' : '#67e8f9')}
          labelResolution={2}
        />
      )}
    </div>
  )
}
