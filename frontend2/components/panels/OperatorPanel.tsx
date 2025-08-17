"use client";
import { apiPost } from "../../lib/api";
import { isOperator } from "../../lib/auth";
import { useRef, useState } from "react";

export function OperatorPanel() {
  const [allowed] = useState<boolean>(isOperator());
  const lastRef = useRef<{ key: string; t: number } | null>(null);

  const needConfirm = (action: string) => /kill|throttle/i.test(action);

  const twoStepOk = (key: string) => {
    const now = Date.now();
    const last = lastRef.current;
    if (last && last.key === key && now - last.t < 5000) { lastRef.current = null; return true; }
    lastRef.current = { key, t: now };
    return false;
  };

  const onAction = async (module: "MEV" | "ARB", action: string, args: any = {}) => {
    if (!allowed) return;
    const key = `${module}:${action}`;
    if (needConfirm(action) && !twoStepOk(key)) return;
    if (needConfirm(action) && !window.confirm(`Confirm ${key}?`)) return;
    await apiPost(`/control/${module.toLowerCase()}:${action}`, args);
  };

  if (!allowed) return <div className="text-sm text-zinc-500">Viewer mode. Operator controls hidden.</div>;

  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="p-4 border border-zinc-800 rounded">
        <h3 className="font-semibold mb-2">MEV Agent</h3>
        <div className="flex gap-2">
          <button className="btn" onClick={() => onAction("MEV", "start")}>Start</button>
          <button className="btn" onClick={() => onAction("MEV", "stop")}>Stop</button>
          <button className="btn" onClick={() => onAction("MEV", "restart")}>Restart</button>
          <button className="btn btn-danger" onClick={() => onAction("MEV", "kill")}>KILL</button>
        </div>
      </div>
      <div className="p-4 border border-zinc-800 rounded">
        <h3 className="font-semibold mb-2">Arbitrage Agent</h3>
        <div className="flex gap-2">
          <button className="btn" onClick={() => onAction("ARB", "start")}>Start</button>
          <button className="btn" onClick={() => onAction("ARB", "stop")}>Stop</button>
          <button className="btn" onClick={() => onAction("ARB", "restart")}>Restart</button>
        </div>
      </div>
      <div className="p-4 border border-zinc-800 rounded col-span-2">
        <h3 className="font-semibold mb-2">Policy & Hedge</h3>
        <div className="flex gap-3">
          <button className="btn" onClick={() => onAction("MEV", "policy", { ev_min: 0.0008, tip_ladder: [0.5, 0.7, 0.85, 0.95] })}>Set Policy</button>
          <button className="btn" onClick={() => onAction("MEV", "hedge", { first_ms: 10, delta_ms: 8, third: true })}>W‑Hedge</button>
          <button className="btn" onClick={() => onAction("MEV", "throttle", { land_min: 0.55, neg_ev_max: 0.01 })}>SLO Guard</button>
          <button className="btn" onClick={() => onAction("MEV", "calibrate", { hours: 6 })}>Calibrate p_land</button>
        </div>
      </div>
    </div>
  );
}