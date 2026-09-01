import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { NavLink } from 'react-router-dom'
import { api } from '../lib/api'
import { useDashboardSocket } from '../lib/ws'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/markets', label: 'Markets' },
  { to: '/strategies', label: 'Strategies' },
  { to: '/trades', label: 'Trades' },
  { to: '/gold', label: 'Gold' },
]

// Locking in a hit profit target is good news, so it gets its own (emerald,
// not red) treatment -- a genuine loss breach and a manual toggle both read
// as neutral/cautionary "trading is halted" states instead.
const BANNER_STYLE: Record<string, { wrap: string; icon: string }> = {
  profit_target: { wrap: 'bg-emerald-950 text-emerald-300 border-emerald-900', icon: '🎯' },
  daily_loss: { wrap: 'bg-red-950 text-red-300 border-red-900', icon: '⛔' },
  manual: { wrap: 'bg-amber-950 text-amber-300 border-amber-900', icon: '✋' },
  // Deliberately the loudest of the four: unlike the daily halts, this one
  // spans days and does not clear on its own at the next day boundary.
  max_drawdown: { wrap: 'bg-red-900 text-red-100 border-red-600', icon: '🛑' },
}

export function Header() {
  const qc = useQueryClient()
  const { snapshot, connected } = useDashboardSocket()
  const [error, setError] = useState<string | null>(null)

  const modeQuery = useQuery({ queryKey: ['mode'], queryFn: api.mode, refetchInterval: 5000 })

  const setModeMutation = useMutation({
    mutationFn: (mode: 'paper' | 'live') => api.setMode(mode),
    onSuccess: () => {
      setError(null)
      qc.invalidateQueries({ queryKey: ['mode'] })
    },
    onError: (e: Error) => setError(e.message),
  })

  const killSwitchMutation = useMutation({
    mutationFn: (on: boolean) => api.killSwitch(on),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['mode'] }),
  })

  const mode = modeQuery.data?.mode ?? 'paper'
  const killSwitchOn = modeQuery.data?.kill_switch ?? false
  const killSwitchReason = modeQuery.data?.kill_switch_reason ?? null

  const banner = killSwitchOn && killSwitchReason ? BANNER_STYLE[killSwitchReason.type] : null

  return (
    <header className="border-b border-neutral-800 bg-neutral-950/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-6">
        <div className="font-semibold text-lg tracking-tight text-neutral-100">
          crypto<span className="text-emerald-400">algo</span>
        </div>

        <nav className="flex gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  isActive ? 'bg-neutral-800 text-neutral-50' : 'text-neutral-400 hover:text-neutral-200'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          {snapshot && (
            <span className="text-xs text-neutral-500">
              equity{' '}
              <span className="text-neutral-200 font-mono">
                ${snapshot.equity.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </span>
            </span>
          )}
          <span className={`h-2 w-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-neutral-600'}`} title="live feed" />

          <div className="flex rounded-md border border-neutral-700 overflow-hidden text-sm">
            <button
              className={`px-3 py-1.5 font-medium ${mode === 'paper' ? 'bg-emerald-600 text-white' : 'bg-neutral-900 text-neutral-400 hover:text-neutral-200'}`}
              onClick={() => setModeMutation.mutate('paper')}
            >
              Paper
            </button>
            <button
              className={`px-3 py-1.5 font-medium ${mode === 'live' ? 'bg-red-600 text-white' : 'bg-neutral-900 text-neutral-400 hover:text-neutral-200'}`}
              onClick={() => {
                if (window.confirm('Switch to LIVE trading? Real orders with real funds will be placed.')) {
                  setModeMutation.mutate('live')
                }
              }}
            >
              Live
            </button>
          </div>

          <button
            className={`px-3 py-1.5 rounded-md text-sm font-semibold border ${
              killSwitchOn
                ? 'bg-red-600 border-red-500 text-white'
                : 'border-neutral-700 text-neutral-400 hover:text-red-400 hover:border-red-500'
            }`}
            onClick={() => killSwitchMutation.mutate(!killSwitchOn)}
            title="Immediately flattens all positions and halts trading"
          >
            {killSwitchOn ? 'KILL SWITCH: ON' : 'Kill Switch'}
          </button>
        </div>
      </div>
      {killSwitchOn && killSwitchReason && banner && (
        <div className={`text-xs px-4 py-1.5 border-t flex items-center gap-2 ${banner.wrap}`}>
          <span>{banner.icon}</span>
          <span className="font-medium">{killSwitchReason.message}</span>
          <span className="font-mono opacity-80">
            ({killSwitchReason.pnl_usd >= 0 ? '+' : ''}
            {killSwitchReason.pnl_usd.toFixed(2)} USD, {(killSwitchReason.pnl_pct * 100).toFixed(1)}% today)
          </span>
        </div>
      )}
      {error && <div className="bg-red-950 text-red-300 text-xs px-4 py-1.5 border-t border-red-900">{error}</div>}
    </header>
  )
}
