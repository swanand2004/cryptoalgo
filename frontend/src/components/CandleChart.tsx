import { useEffect, useRef } from 'react'
import {
  createChart,
  createSeriesMarkers,
  ColorType,
  CandlestickSeries,
  LineSeries,
  LineStyle,
  type IChartApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts'
import type { Candle } from '../lib/api'

export interface PriceLine {
  price: number
  color: string
  title: string
  dashed?: boolean
  // Bar-open time (unix seconds) the level starts at. Set: drawn as a segment
  // from that bar to the right edge, the way a trade's levels exist only from
  // its entry on. Unset: a full-width horizontal line.
  from?: number
}

export interface ChartMarker {
  time: number
  position: 'aboveBar' | 'belowBar'
  shape: 'arrowUp' | 'arrowDown' | 'circle'
  color: string
  text: string
}

const NO_LINES: PriceLine[] = []
const NO_MARKERS: ChartMarker[] = []
const EXTEND_BARS = 8 // how far segments run into the empty space right of the last bar

export function CandleChart({
  candles,
  height = 360,
  lines = NO_LINES,
  markers = NO_MARKERS,
}: {
  candles: Candle[]
  height?: number
  lines?: PriceLine[]
  markers?: ChartMarker[]
}) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const chart: IChartApi = createChart(containerRef.current, {
      height,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#9aa7b2' },
      grid: { vertLines: { color: '#1b222b' }, horzLines: { color: '#1b222b' } },
      timeScale: { timeVisible: true, borderColor: '#2a323c' },
      rightPriceScale: { borderColor: '#2a323c' },
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#34d399',
      downColor: '#f87171',
      borderVisible: false,
      wickUpColor: '#34d399',
      wickDownColor: '#f87171',
    })

    const resize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth })
    }
    resize()
    window.addEventListener('resize', resize)

    if (candles.length > 0) {
      series.setData(
        candles.map((c) => ({
          time: c.time as Time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
      )
    }

    const first = candles.length ? candles[0].time : 0
    const last = candles.length ? candles[candles.length - 1].time : 0
    const step = candles.length > 1 ? last - candles[candles.length - 2].time : 0

    for (const line of lines) {
      const style = line.dashed ? LineStyle.Dashed : LineStyle.Solid
      if (line.from === undefined || !candles.length || step <= 0) {
        series.createPriceLine({
          price: line.price, color: line.color, title: line.title,
          lineWidth: 1, lineStyle: style, axisLabelVisible: true,
        })
        continue
      }
      // Snap the start onto the visible range, then run to the last bar and on
      // past it. Times must strictly increase, so the start is only added when
      // it is before the last bar.
      const start = Math.max(first, Math.min(line.from, last))
      const times = start < last ? [start, last] : [last]
      for (let k = 1; k <= EXTEND_BARS; k++) times.push(last + k * step)
      const segment = chart.addSeries(LineSeries, {
        color: line.color,
        lineWidth: 1,
        lineStyle: style,
        title: line.title,
        priceLineVisible: false,
        lastValueVisible: true,
        crosshairMarkerVisible: false,
      })
      segment.setData(times.map((t) => ({ time: t as Time, value: line.price })))
    }

    if (markers.length && candles.length) {
      const sorted: SeriesMarker<Time>[] = [...markers]
        .filter((m) => m.time >= first && m.time <= last)
        .sort((a, b) => a.time - b.time)
        .map((m) => ({ ...m, time: m.time as Time }))
      createSeriesMarkers(series, sorted)
    }

    if (candles.length > 0) chart.timeScale().fitContent()

    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
  }, [candles, height, lines, markers])

  return <div ref={containerRef} className="w-full" />
}
