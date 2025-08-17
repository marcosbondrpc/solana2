"use client";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef } from "react";

export function VirtualTable({ rows }: { rows: any[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 20,
  });
  return (
    <div ref={parentRef} className="h-[360px] overflow-auto border border-zinc-800 rounded">
      <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: "relative" }}>
        {rowVirtualizer.getVirtualItems().map((v) => (
          <div
            key={v.key}
            style={{ position: "absolute", top: 0, left: 0, width: "100%", transform: `translateY(${v.start}px)` }}
            className="px-3 py-1 text-xs border-b border-zinc-900"
          >
            {JSON.stringify(rows[v.index])}
          </div>
        ))}
      </div>
    </div>
  );
}