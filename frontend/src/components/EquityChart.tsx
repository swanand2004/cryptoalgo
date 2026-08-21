import { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries, type IChartApi, type Time } from 'lightweight-charts'
import type { EquityPoint } from '../lib/api'

function toTime(p: EquityPoint): number {
  if (p.time) return p.time
  if (p.timestamp) return Math.floor(new Date(p.timestamp).getTime() / 1000)
  return 0
}

export function EquityChart({ points, height = 260 }: { points: EquityPoint[]; height?: number }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      height,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#9aa7b2' },
      grid: { vertLines: { color: '#1b222b' }, horzLines: { color: '#1b222b' } },
      timeScale: { timeVisible: true, borderColor: '#2a323c' },
      rightPriceScale: { borderColor: '#2a323c' },
    })
    const series = chart.addSeries(LineSeries, { color: '#34d399', lineWidth: 2 })
    chartRef.current = chart

    const resize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth })
    }
    resize()
    window.addEventListener('resize', resize)

    if (points.length > 0) {
      series.setData(
        points.map((p) => ({ time: toTime(p) as Time, value: p.equity })).sort((a, b) => (a.time as number) - (b.time as number))
      )
      chart.timeScale().fitContent()
    }

    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, height])

  return <div ref={containerRef} className="w-full" />
}
