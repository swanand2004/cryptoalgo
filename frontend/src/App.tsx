import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import Dashboard from './pages/Dashboard'
import Markets from './pages/Markets'
import Strategies from './pages/Strategies'
import Trades from './pages/Trades'
import Gold from './pages/Gold'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="markets" element={<Markets />} />
        <Route path="strategies" element={<Strategies />} />
        <Route path="trades" element={<Trades />} />
        <Route path="gold" element={<Gold />} />
      </Route>
    </Routes>
  )
}
