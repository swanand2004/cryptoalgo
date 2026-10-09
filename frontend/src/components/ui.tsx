import type { ReactNode } from 'react'
import { errorText } from '../lib/hooks'

// Shared building blocks so every page loads, fails and empties the same way.

export function Panel({
  title,
  right,
  children,
  className = '',
  bodyClassName = 'p-4',
}: {
  title?: ReactNode
  right?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={`min-w-0 rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {(title || right) && (
        <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-neutral-800 px-4 py-2.5">
          {title && <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-300">{title}</h2>}
          {right}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}

export function Skeleton({ className = 'h-24' }: { className?: string }) {
  return <div aria-hidden="true" className={`animate-pulse rounded-lg bg-neutral-800/60 motion-reduce:animate-none ${className}`} />
}

/** Loading placeholder that also tells screen readers what is happening. */
export function Loading({ className = 'h-24', label = 'Loading' }: { className?: string; label?: string }) {
  return (
    <div role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      <Skeleton className={className} />
    </div>
  )
}

export function ErrorState({ what, error, onRetry }: { what: string; error?: unknown; onRetry?: () => void }) {
  const detail = error ? errorText(error) : ''
  return (
    <div role="alert" className="rounded-lg border border-red-900/70 bg-red-950/30 px-3 py-2.5 text-sm text-red-300">
      <div className="font-medium">Couldn&rsquo;t load {what}.</div>
      {detail && <div className="mt-0.5 break-words text-xs text-red-300/80">{detail}</div>}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 rounded-md border border-red-800 px-2.5 py-1 text-xs font-medium text-red-200 hover:bg-red-900/40"
        >
          Try again
        </button>
      )}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-6 text-center text-sm text-neutral-400">{children}</div>
}

/** A label/value pair, as used inside the phone cards. */
export function Field({
  label,
  children,
  className = '',
  align = 'left',
}: {
  label: string
  children: ReactNode
  className?: string
  align?: 'left' | 'right'
}) {
  return (
    <div className={`min-w-0 ${align === 'right' ? 'text-right' : ''} ${className}`}>
      <div className="text-[11px] uppercase tracking-wide text-neutral-400">{label}</div>
      <div className="truncate font-mono text-sm tabular-nums">{children}</div>
    </div>
  )
}
