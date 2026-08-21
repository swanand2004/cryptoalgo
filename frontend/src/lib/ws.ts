import { useEffect, useRef, useState } from 'react'

export interface DashboardSnapshot {
  type: 'snapshot'
  mode: string
  equity: number
  prices: Record<string, number | null>
  positions: {
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
  }[]
  system_status: string
}

export function useDashboardSocket() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout>
    let staleTimer: ReturnType<typeof setTimeout>

    function connect() {
      if (cancelled) return
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/dashboard`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        clearTimeout(staleTimer)
      }
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as DashboardSnapshot
          setSnapshot(data)
        } catch {
          // ignore malformed frame
        }
      }
      ws.onclose = () => {
        setConnected(false)
        // Brief reconnect blips (a dropped frame, a dev-server proxy
        // hiccup) happen periodically and self-heal within the 2s retry
        // below -- clearing the snapshot on every single one of those made
        // LTP/positions flicker to "--" and back instead of holding
        // steady, which is worse than briefly showing a few-second-old
        // price. Only actually drop the snapshot (falling back to callers'
        // slower REST polls) if the socket stays down for a real stretch.
        clearTimeout(staleTimer)
        staleTimer = setTimeout(() => setSnapshot(null), 8000)
        if (!cancelled) retryTimer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      cancelled = true
      clearTimeout(retryTimer)
      clearTimeout(staleTimer)
      wsRef.current?.close()
    }
  }, [])

  return { snapshot, connected }
}
