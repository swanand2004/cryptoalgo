import { Outlet } from 'react-router-dom'
import { Header } from './Header'

export function Layout() {
  return (
    <div className="min-h-screen">
      <Header />
      {/* No max-width/centering here -- each page applies its own (see
          Dashboard/Markets/Strategies vs. Trades), so a page that needs to
          be wider (Trades' 14-column table) can be, without a fragile
          "break out of an already-centered parent" CSS trick. */}
      <main className="px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
