import type { Config } from 'tailwindcss'
export default {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  theme: { extend: { colors: { brand: { 600: "#6f5cff" } } } },
  plugins: []
} satisfies Config
