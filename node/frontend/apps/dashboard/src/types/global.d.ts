declare module "*.svg" { const src: string; export default src; }
declare module "*.png" { const src: string; export default src; }
declare module "*.jpg" { const src: string; export default src; }
declare module "*.jpeg" { const src: string; export default src; }
declare module "*.webp" { const src: string; export default src; }
declare module "*?worker" { const WorkerConstructor: { new (): Worker }; export default WorkerConstructor; }
declare module "*.worker.ts" { const WorkerConstructor: { new (): Worker }; export default WorkerConstructor; }
declare module "react-virtualized-auto-sizer";
declare module "*.module.css";
declare module "*.module.scss";