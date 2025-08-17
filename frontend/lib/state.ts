import { create } from "zustand";

type SysState = {
  latest: any[];
  pushLatest: (x: any) => void;
};

export const useSys = create<SysState>((set) => ({
  latest: [],
  pushLatest: (x) =>
    set((s) => ({
      latest: s.latest.length > 1000 ? [...s.latest.slice(-900), x] : [...s.latest, x],
    })),
}));