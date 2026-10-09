import { useEffect, useRef, useState } from 'react'
import type { Position } from './api'

export interface DashboardSnapshot {
  type: 'snapshot'
  mode: string
  equity: number | null // null while Delta's balance can't be read
  prices: Record<string, number | null>
  // Same shape as GET /api/trading/positions (see Position in api.ts).
  positions: Position[]
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
