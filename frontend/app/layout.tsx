import "../styles/globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-black text-white">
        <nav className="px-6 py-3 border-b border-zinc-800 sticky top-0 bg-black/70 backdrop-blur">
          <div className="flex items-center gap-4 text-sm">
            <a href="/" className="font-semibold text-brand-600">Legendary Node</a>
            <a href="/mev" className="text-yellow-400 font-bold">MEV</a>
            <a href="/realtime">Realtime</a>
            <a href="/control">Control</a>
            <a href="/datasets">Datasets</a>
            <a href="/training">Training</a>
            <a href="/models">Models</a>
            <a href="/analytics">Analytics</a>
          </div>
        </nav>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
