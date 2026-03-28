export function Navbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-zinc-800/70 bg-[#0D0F14]/70 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.8)]" />
          <span className="text-sm font-semibold tracking-wide text-zinc-100">
            DATA ENGINEERING AGENT
          </span>
        </div>
        <span className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-400">
          AutoML Workspace
        </span>
      </div>
    </header>
  );
}
