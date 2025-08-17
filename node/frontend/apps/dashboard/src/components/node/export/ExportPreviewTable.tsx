import React from 'react';
import { Card } from '../../ui/card';
import { Eye } from 'lucide-react';

interface ExportPreviewTableProps {
  data: any[];
}

export const ExportPreviewTable: React.FC<ExportPreviewTableProps> = ({ data }) => {
  if (!data || data.length === 0) return null;

  const columns = Object.keys(data[0]);
  const previewRows = data.slice(0, 10);

  return (
    <Card className="p-6 bg-gray-800/30 border-gray-700/50">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500/10 to-pink-500/10">
          <Eye className="w-5 h-5 text-purple-400" />
        </div>
        <h3 className="text-lg font-medium text-white">Data Preview</h3>
        <span className="text-sm text-gray-400">First 10 rows</span>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {previewRows.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors"
              >
                {columns.map((col) => (
                  <td
                    key={col}
                    className="px-3 py-2 text-gray-300 font-mono text-xs"
                  >
                    {typeof row[col] === 'number'
                      ? row[col].toFixed(4)
                      : row[col]?.toString() || '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};