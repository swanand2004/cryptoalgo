import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries, type IChartApi, type Time } from 'lightweight-charts'
import type { Candle } from '../lib/api'

export function CandleChart({ candles, height = 360 }: { candles: Candle[]; height?: number }) {
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
      chart.timeScale().fitContent()
    }

    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
  }, [candles, height])

  return <div ref={containerRef} className="w-full" />
}
