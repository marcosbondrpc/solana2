import { Card } from '../../components/ui/Card';

export default function OpsHome() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      <Card title="System Health">
        <div className="text-sm text-zinc-500">Live status and KPIs</div>
      </Card>
      <Card title="Throughput">
        <div className="h-24 rounded bg-zinc-100 dark:bg-zinc-800" />
      </Card>
      <Card title="Alerts">
        <ul className="text-sm list-disc pl-4">
          <li>No critical alerts</li>
        </ul>
      </Card>
    </div>
  );
}