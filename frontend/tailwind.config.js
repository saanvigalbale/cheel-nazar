/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        recon: {
          bg: '#0a0d14',
          panel: '#111726',
          border: '#1e293b',
          accent: '#06b6d4',
          warning: '#f59e0b',
          danger: '#ef4444',
          success: '#10b981',
          tactical: '#38bdf8',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
