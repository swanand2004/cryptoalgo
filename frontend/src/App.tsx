import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import Dashboard from './pages/Dashboard'
import Markets from './pages/Markets'
import Trades from './pages/Trades'
import Gold from './pages/Gold'
import Analyst from './pages/Analyst'
import Options from './pages/Options'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="markets" element={<Markets />} />
        <Route path="trades" element={<Trades />} />
        <Route path="gold" element={<Gold />} />
        <Route path="analyst" element={<Analyst />} />
        <Route path="options" element={<Options />} />
      </Route>
    </Routes>
  )
}
