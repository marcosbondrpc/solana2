"use client";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useEffect, useRef } from "react";

export function StreamChart({ points }: { points: [number, number][] }) {
  const ref = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot>();
  useEffect(() => {
    if (!ref.current) return;
    if (!plot.current) {
      plot.current = new uPlot(
        {
          title: "Profit (SOL) vs Time",
          width: ref.current.clientWidth,
          height: 240,
          series: [{}, { label: "Profit", points: { show: false } }],
          axes: [{}, { label: "SOL" }],
        },
        [points.map((p) => p[0]), points.map((p) => p[1])],
        ref.current
      );
      const onResize = () =>
        plot.current!.setSize({ width: ref.current!.clientWidth, height: 240 });
      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    } else {
      plot.current.setData([points.map((p) => p[0]), points.map((p) => p[1])]);
    }
  }, [points]);
  return <div ref={ref} className="w-full" />;
}