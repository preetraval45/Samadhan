import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Brand colors matching logo
        brand: {
          cyan: '#00d4ff',
          blue: '#0099ff',
          'dark-blue': '#0066ff',
          orange: '#ff6b00',
          'light-orange': '#ff9500',
          dark: '#0a0e27',
          'dark-secondary': '#1a1f3a',
        },
        // UI colors
        background: {
          DEFAULT: '#0a0e27',
          secondary: '#1a1f3a',
          tertiary: '#252b45',
        },
        text: {
          primary: '#ffffff',
          secondary: '#a0aec0',
          accent: '#00d4ff',
        }
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #00d4ff 0%, #0099ff 50%, #0066ff 100%)',
        'gradient-accent': 'linear-gradient(135deg, #ff6b00 0%, #ff9500 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}

export default config
