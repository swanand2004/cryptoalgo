import { useEffect, useState } from 'react'

/** The request's error text, without a whole HTML error page in it. */
export function errorText(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err ?? '')
  const firstLine = raw.split('\n')[0].replace(/<[^>]*>/g, '').trim()
  return firstLine.length > 160 ? `${firstLine.slice(0, 160)}...` : firstLine
}

/** True while the viewport matches the media query (e.g. phone width). */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(query).matches : false
  )
  useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia(query)
    const onChange = () => setMatches(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])
  return matches
}

/** Below Tailwind's `sm` breakpoint: a phone held upright. */
export function useIsPhone(): boolean {
  return useMediaQuery('(max-width: 639px)')
}
