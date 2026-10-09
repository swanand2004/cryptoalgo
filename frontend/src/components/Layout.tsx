import { Outlet } from 'react-router-dom'
import { Header } from './Header'

export function Layout() {
  return (
    <div className="min-h-screen">
      <Header />
      {/* No max-width here -- each page sets its own (Trades is wider for
          its desktop table). 16px side gutter on every screen size. */}
      <main className="px-4 py-4 sm:py-6">
        <Outlet />
      </main>
    </div>
  )
}
