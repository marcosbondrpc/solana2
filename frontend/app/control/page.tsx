"use client";
import { OperatorPanel } from "../../components/panels/OperatorPanel";

export default function ControlPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Control Plane</h1>
      <OperatorPanel />
    </div>
  );
}