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
]

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
      {error && <div className="bg-red-950 text-red-300 text-xs px-4 py-1.5 border-t border-red-900">{error}</div>}
    </header>
  )
}
