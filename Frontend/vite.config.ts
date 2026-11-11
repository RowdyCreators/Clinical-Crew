import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During development, API calls are proxied to the FastAPI backend (see README).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8010', changeOrigin: true },
      '/upload': { target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8010', changeOrigin: true },
      '/static': { target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8010', changeOrigin: true },
    },
  },
})
