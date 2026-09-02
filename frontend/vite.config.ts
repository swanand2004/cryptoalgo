import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Without an explicit host, Node's `listen(port, 'localhost')` binds to
    // whichever address 'localhost' resolves to FIRST -- on this machine
    // that was only the IPv6 loopback ([::1]), so anything resolving
    // "localhost"/127.0.0.1 to IPv4 (many browsers/OS network stacks do)
    // got a flat connection-refused on every page. true binds both
    // 0.0.0.0 and :: so it answers either way.
    host: true,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
