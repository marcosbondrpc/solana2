/**
 * ClickHouse Query Builder Interface
 * Advanced SQL editor with real-time query execution and visualization
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tabs,
  Space,
  Select,
  Input,
  Alert,
  Tag,
  Tooltip,
  Dropdown,
  Menu,
  Modal,
  Form,
  Switch,
  InputNumber,
  Divider,
  Typography,
  Statistic,
  Timeline,
  Badge,
  message,
  notification,
  Drawer,
  Tree,
  Empty,
  Spin,
  AutoComplete,
  Segmented,
} from 'antd';
import {
  DatabaseOutlined,
  PlayCircleOutlined,
  SaveOutlined,
  HistoryOutlined,
  DownloadOutlined,
  TableOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  HeatMapOutlined,
  SettingOutlined,
  FormatPainterOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  CopyOutlined,
  FileTextOutlined,
  QuestionCircleOutlined,
  FolderOutlined,
  FileOutlined,
  SearchOutlined,
  FilterOutlined,
  ExportOutlined,
  CodeOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import MonacoEditor from '@monaco-editor/react';
import { Line, Column, Area, Pie, Scatter, Heatmap } from '@ant-design/plots';
import { apiService, ClickHouseQuery } from '../../services/api-service';
import { useMEVStore } from '../../stores/mev-store';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;
const { DirectoryTree } = Tree;

interface QueryResult {
  columns: Array<{ name: string; type: string }>;
  rows: any[];
  rowCount: number;
  executionTime: number;
  bytesRead: number;
  timestamp: number;
}

interface SavedQuery {
  id: string;
  name: string;
  query: string;
  description?: string;
  tags: string[];
  createdAt: number;
  lastRun?: number;
  runCount: number;
}

interface TableSchema {
  database: string;
  table: string;
  columns: Array<{
    name: string;
    type: string;
    nullable: boolean;
    default?: string;
    comment?: string;
  }>;
  engine: string;
  rowCount?: number;
  dataSize?: string;
}

interface QueryHistory {
  id: string;
  query: string;
  executionTime: number;
  rowCount: number;
  status: 'success' | 'error';
  error?: string;
  timestamp: number;
}

export const ClickHouseQueryBuilder: React.FC = () => {
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [tables, setTables] = useState<string[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [tableSchema, setTableSchema] = useState<TableSchema | null>(null);
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState('default');
  const [databases, setDatabases] = useState<string[]>(['default']);
  const [visualizationType, setVisualizationType] = useState<'table' | 'chart'>('table');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area' | 'pie' | 'scatter' | 'heatmap'>('line');
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [schemaDrawerVisible, setSchemaDrawerVisible] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'json' | 'parquet'>('csv');
  const [autoComplete, setAutoComplete] = useState(true);
  const [queryLimit, setQueryLimit] = useState(1000);
  
  const editorRef = useRef<any>(null);
  const [saveForm] = Form.useForm();

  // Fetch tables on mount
  useEffect(() => {
    fetchTables();
    loadSavedQueries();
    loadQueryHistory();
  }, [selectedDatabase]);

  // Fetch available tables
  const fetchTables = async () => {
    try {
      const tableList = await apiService.getClickHouseTables();
      setTables(tableList);
    } catch (error) {
      console.error('Failed to fetch tables:', error);
      message.error('Failed to load tables');
    }
  };

  // Load saved queries from localStorage
  const loadSavedQueries = () => {
    const saved = localStorage.getItem('clickhouse_saved_queries');
    if (saved) {
      setSavedQueries(JSON.parse(saved));
    }
  };

  // Load query history from localStorage
  const loadQueryHistory = () => {
    const history = localStorage.getItem('clickhouse_query_history');
    if (history) {
      setQueryHistory(JSON.parse(history).slice(0, 50)); // Keep last 50
    }
  };

  // Execute query
  const executeQuery = async () => {
    if (!query.trim()) {
      message.warning('Please enter a query');
      return;
    }

    setLoading(true);
    const startTime = Date.now();

    try {
      const queryRequest: ClickHouseQuery = {
        query: query.trim(),
        database: selectedDatabase,
        format: 'JSON',
        timeout_ms: 30000,
        max_rows: queryLimit,
      };

      const response = await apiService.executeClickHouseQuery(queryRequest);
      const executionTime = Date.now() - startTime;

      const result: QueryResult = {
        columns: response.meta || [],
        rows: response.data || [],
        rowCount: response.rows || 0,
        executionTime,
        bytesRead: response.statistics?.bytes_read || 0,
        timestamp: Date.now(),
      };

      setQueryResult(result);

      // Add to history
      const historyEntry: QueryHistory = {
        id: `${Date.now()}`,
        query: query.trim(),
        executionTime,
        rowCount: result.rowCount,
        status: 'success',
        timestamp: Date.now(),
      };

      const newHistory = [historyEntry, ...queryHistory.slice(0, 49)];
      setQueryHistory(newHistory);
      localStorage.setItem('clickhouse_query_history', JSON.stringify(newHistory));

      notification.success({
        message: 'Query Executed',
        description: `Returned ${result.rowCount} rows in ${executionTime}ms`,
        placement: 'bottomRight',
      });
    } catch (error: any) {
      const historyEntry: QueryHistory = {
        id: `${Date.now()}`,
        query: query.trim(),
        executionTime: Date.now() - startTime,
        rowCount: 0,
        status: 'error',
        error: error.message,
        timestamp: Date.now(),
      };

      const newHistory = [historyEntry, ...queryHistory.slice(0, 49)];
      setQueryHistory(newHistory);
      localStorage.setItem('clickhouse_query_history', JSON.stringify(newHistory));

      notification.error({
        message: 'Query Failed',
        description: error.message,
        placement: 'bottomRight',
      });
    } finally {
      setLoading(false);
    }
  };

  // Save query
  const handleSaveQuery = (values: any) => {
    const newQuery: SavedQuery = {
      id: `${Date.now()}`,
      name: values.name,
      query: query,
      description: values.description,
      tags: values.tags || [],
      createdAt: Date.now(),
      runCount: 0,
    };

    const updated = [...savedQueries, newQuery];
    setSavedQueries(updated);
    localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updated));

    message.success('Query saved successfully');
    setSaveModalVisible(false);
    saveForm.resetFields();
  };

  // Load saved query
  const loadSavedQuery = (savedQuery: SavedQuery) => {
    setQuery(savedQuery.query);
    
    // Update run count
    const updated = savedQueries.map(q => 
      q.id === savedQuery.id 
        ? { ...q, lastRun: Date.now(), runCount: q.runCount + 1 }
        : q
    );
    setSavedQueries(updated);
    localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updated));
  };

  // Delete saved query
  const deleteSavedQuery = (id: string) => {
    Modal.confirm({
      title: 'Delete Query',
      content: 'Are you sure you want to delete this saved query?',
      onOk: () => {
        const updated = savedQueries.filter(q => q.id !== id);
        setSavedQueries(updated);
        localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updated));
        message.success('Query deleted');
      },
    });
  };

  // Format SQL
  const formatSQL = () => {
    // Simple SQL formatter - in production, use a proper SQL formatter library
    const formatted = query
      .replace(/\s+/g, ' ')
      .replace(/,/g, ',\n  ')
      .replace(/SELECT/gi, 'SELECT\n  ')
      .replace(/FROM/gi, '\nFROM')
      .replace(/WHERE/gi, '\nWHERE')
      .replace(/GROUP BY/gi, '\nGROUP BY')
      .replace(/ORDER BY/gi, '\nORDER BY')
      .replace(/LIMIT/gi, '\nLIMIT');
    
    setQuery(formatted);
  };

  // Export results
  const exportResults = async () => {
    if (!queryResult) {
      message.warning('No results to export');
      return;
    }

    try {
      const blob = await apiService.exportData('query_results', exportFormat as any, {
        start: Date.now() - 86400000,
        end: Date.now(),
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `query_results_${Date.now()}.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);

      message.success('Results exported successfully');
    } catch (error) {
      message.error('Failed to export results');
    }
  };

  // Get chart data
  const getChartData = useCallback(() => {
    if (!queryResult || queryResult.rows.length === 0) return [];

    // Assume first column is X-axis, second is Y-axis for simple charts
    const xKey = queryResult.columns[0]?.name;
    const yKey = queryResult.columns[1]?.name;

    return queryResult.rows.map(row => ({
      x: row[xKey],
      y: row[yKey],
      ...row,
    }));
  }, [queryResult]);

  // Render chart
  const renderChart = () => {
    const data = getChartData();
    if (data.length === 0) return <Empty description="No data to visualize" />;

    const commonConfig = {
      data,
      xField: 'x',
      yField: 'y',
      smooth: true,
      animation: {
        appear: {
          animation: 'path-in',
          duration: 1000,
        },
      },
    };

    switch (chartType) {
      case 'line':
        return <Line {...commonConfig} />;
      case 'bar':
        return <Column {...commonConfig} />;
      case 'area':
        return <Area {...commonConfig} />;
      case 'pie':
        return <Pie data={data} angleField="y" colorField="x" />;
      case 'scatter':
        return <Scatter {...commonConfig} />;
      case 'heatmap':
        return <Heatmap {...commonConfig} xField="x" yField="x" colorField="y" />;
      default:
        return null;
    }
  };

  // Table columns
  const getTableColumns = () => {
    if (!queryResult) return [];

    return queryResult.columns.map(col => ({
      title: (
        <Space>
          {col.name}
          <Tag color="blue" style={{ fontSize: 10 }}>
            {col.type}
          </Tag>
        </Space>
      ),
      dataIndex: col.name,
      key: col.name,
      ellipsis: true,
      width: 150,
    }));
  };

  // Query templates
  const queryTemplates = [
    {
      name: 'MEV Opportunities',
      query: `SELECT 
  timestamp,
  dex_a,
  dex_b,
  expected_profit,
  confidence
FROM mev.opportunities
WHERE timestamp > now() - INTERVAL 1 HOUR
ORDER BY expected_profit DESC
LIMIT 100`,
    },
    {
      name: 'Bundle Success Rate',
      query: `SELECT 
  toStartOfMinute(timestamp) as minute,
  countIf(status = 'landed') / count() as success_rate
FROM mev.bundles
WHERE timestamp > now() - INTERVAL 1 DAY
GROUP BY minute
ORDER BY minute`,
    },
    {
      name: 'Top Profitable Tokens',
      query: `SELECT 
  token,
  sum(profit) as total_profit,
  count() as trade_count,
  avg(profit) as avg_profit
FROM mev.trades
WHERE timestamp > now() - INTERVAL 7 DAY
GROUP BY token
ORDER BY total_profit DESC
LIMIT 20`,
    },
  ];

  return (
    <div className="clickhouse-query-builder">
      {/* Header */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Row align="middle" justify="space-between">
              <Col>
                <Space>
                  <DatabaseOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                  <div>
                    <Title level={4} style={{ margin: 0 }}>
                      ClickHouse Query Builder
                    </Title>
                    <Text type="secondary">
                      Execute and visualize queries on your MEV data
                    </Text>
                  </div>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Select
                    value={selectedDatabase}
                    onChange={setSelectedDatabase}
                    style={{ width: 150 }}
                  >
                    {databases.map(db => (
                      <Option key={db} value={db}>{db}</Option>
                    ))}
                  </Select>
                  <Tooltip title="Auto-complete">
                    <Switch
                      checkedChildren={<CodeOutlined />}
                      unCheckedChildren={<CodeOutlined />}
                      checked={autoComplete}
                      onChange={setAutoComplete}
                    />
                  </Tooltip>
                  <Button
                    icon={<ThunderboltOutlined />}
                    onClick={() => setSchemaDrawerVisible(true)}
                  >
                    Schema
                  </Button>
                  <Button
                    icon={<HistoryOutlined />}
                    onClick={() => {
                      // Show history modal
                    }}
                  >
                    History
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Main Content */}
      <Row gutter={[16, 16]}>
        {/* Query Editor */}
        <Col span={16}>
          <Card
            title={
              <Space>
                <CodeOutlined />
                SQL Editor
                <Dropdown
                  overlay={
                    <Menu>
                      {queryTemplates.map((template, i) => (
                        <Menu.Item
                          key={i}
                          onClick={() => setQuery(template.query)}
                        >
                          {template.name}
                        </Menu.Item>
                      ))}
                    </Menu>
                  }
                >
                  <Button size="small">Templates</Button>
                </Dropdown>
              </Space>
            }
            extra={
              <Space>
                <Button
                  icon={<FormatPainterOutlined />}
                  size="small"
                  onClick={formatSQL}
                >
                  Format
                </Button>
                <Button
                  icon={<SaveOutlined />}
                  size="small"
                  onClick={() => setSaveModalVisible(true)}
                  disabled={!query}
                >
                  Save
                </Button>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={executeQuery}
                  loading={loading}
                  disabled={!query}
                >
                  Execute
                </Button>
              </Space>
            }
          >
            <MonacoEditor
              height="400px"
              language="sql"
              theme="vs-dark"
              value={query}
              onChange={(value) => setQuery(value || '')}
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                wordWrap: 'on',
                automaticLayout: true,
                suggestOnTriggerCharacters: autoComplete,
                quickSuggestions: autoComplete,
              }}
              onMount={(editor) => {
                editorRef.current = editor;
              }}
            />

            {/* Query Options */}
            <Divider />
            <Row gutter={16} align="middle">
              <Col>
                <Space>
                  <Text>Limit:</Text>
                  <InputNumber
                    value={queryLimit}
                    onChange={(val) => setQueryLimit(val || 1000)}
                    min={1}
                    max={100000}
                    style={{ width: 100 }}
                  />
                </Space>
              </Col>
              <Col>
                <Space>
                  <Text>Format:</Text>
                  <Select
                    value={exportFormat}
                    onChange={setExportFormat}
                    style={{ width: 100 }}
                  >
                    <Option value="csv">CSV</Option>
                    <Option value="json">JSON</Option>
                    <Option value="parquet">Parquet</Option>
                  </Select>
                </Space>
              </Col>
            </Row>
          </Card>

          {/* Results */}
          {queryResult && (
            <Card
              style={{ marginTop: 16 }}
              title={
                <Space>
                  <TableOutlined />
                  Results
                  <Tag color="green">{queryResult.rowCount} rows</Tag>
                  <Tag color="blue">{queryResult.executionTime}ms</Tag>
                </Space>
              }
              extra={
                <Space>
                  <Segmented
                    value={visualizationType}
                    onChange={setVisualizationType as any}
                    options={[
                      { label: 'Table', value: 'table', icon: <TableOutlined /> },
                      { label: 'Chart', value: 'chart', icon: <BarChartOutlined /> },
                    ]}
                  />
                  {visualizationType === 'chart' && (
                    <Select
                      value={chartType}
                      onChange={setChartType}
                      style={{ width: 100 }}
                    >
                      <Option value="line">Line</Option>
                      <Option value="bar">Bar</Option>
                      <Option value="area">Area</Option>
                      <Option value="pie">Pie</Option>
                      <Option value="scatter">Scatter</Option>
                      <Option value="heatmap">Heatmap</Option>
                    </Select>
                  )}
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={exportResults}
                  >
                    Export
                  </Button>
                </Space>
              }
            >
              {visualizationType === 'table' ? (
                <Table
                  columns={getTableColumns()}
                  dataSource={queryResult.rows}
                  rowKey={(record, index) => index?.toString() || ''}
                  scroll={{ x: 'max-content' }}
                  pagination={{
                    pageSize: 50,
                    showSizeChanger: true,
                    showTotal: (total) => `Total ${total} rows`,
                  }}
                />
              ) : (
                <div style={{ height: 400 }}>
                  {renderChart()}
                </div>
              )}
            </Card>
          )}
        </Col>

        {/* Sidebar */}
        <Col span={8}>
          {/* Saved Queries */}
          <Card
            title={<Space><SaveOutlined /> Saved Queries</Space>}
            style={{ marginBottom: 16 }}
          >
            {savedQueries.length > 0 ? (
              <Timeline>
                {savedQueries.map((sq) => (
                  <Timeline.Item key={sq.id}>
                    <Card
                      size="small"
                      hoverable
                      onClick={() => loadSavedQuery(sq)}
                      extra={
                        <Button
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSavedQuery(sq.id);
                          }}
                        />
                      }
                    >
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        <Text strong>{sq.name}</Text>
                        {sq.description && (
                          <Text type="secondary" ellipsis>
                            {sq.description}
                          </Text>
                        )}
                        <Space>
                          {sq.tags.map(tag => (
                            <Tag key={tag} color="blue">{tag}</Tag>
                          ))}
                        </Space>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          Run {sq.runCount} times
                        </Text>
                      </Space>
                    </Card>
                  </Timeline.Item>
                ))}
              </Timeline>
            ) : (
              <Empty description="No saved queries" />
            )}
          </Card>

          {/* Query History */}
          <Card
            title={<Space><HistoryOutlined /> Recent Queries</Space>}
          >
            {queryHistory.length > 0 ? (
              <Timeline>
                {queryHistory.slice(0, 5).map((h) => (
                  <Timeline.Item
                    key={h.id}
                    color={h.status === 'success' ? 'green' : 'red'}
                  >
                    <Card
                      size="small"
                      hoverable
                      onClick={() => setQuery(h.query)}
                    >
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        <Paragraph
                          ellipsis={{ rows: 2 }}
                          style={{ marginBottom: 0, fontSize: 12 }}
                        >
                          <code>{h.query}</code>
                        </Paragraph>
                        <Space>
                          <Tag color={h.status === 'success' ? 'green' : 'red'}>
                            {h.status}
                          </Tag>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {h.executionTime}ms
                          </Text>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {h.rowCount} rows
                          </Text>
                        </Space>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {new Date(h.timestamp).toLocaleTimeString()}
                        </Text>
                        {h.error && (
                          <Alert
                            message={h.error}
                            type="error"
                            showIcon
                            style={{ fontSize: 11 }}
                          />
                        )}
                      </Space>
                    </Card>
                  </Timeline.Item>
                ))}
              </Timeline>
            ) : (
              <Empty description="No query history" />
            )}
          </Card>
        </Col>
      </Row>

      {/* Save Query Modal */}
      <Modal
        title="Save Query"
        visible={saveModalVisible}
        onCancel={() => setSaveModalVisible(false)}
        footer={null}
      >
        <Form
          form={saveForm}
          layout="vertical"
          onFinish={handleSaveQuery}
        >
          <Form.Item
            name="name"
            label="Query Name"
            rules={[{ required: true, message: 'Please enter a name' }]}
          >
            <Input placeholder="e.g., Daily MEV Performance" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea rows={3} placeholder="Describe what this query does..." />
          </Form.Item>

          <Form.Item
            name="tags"
            label="Tags"
          >
            <Select mode="tags" placeholder="Add tags...">
              <Option value="mev">MEV</Option>
              <Option value="performance">Performance</Option>
              <Option value="analytics">Analytics</Option>
              <Option value="monitoring">Monitoring</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                Save
              </Button>
              <Button onClick={() => setSaveModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Schema Drawer */}
      <Drawer
        title="Database Schema"
        placement="left"
        width={400}
        visible={schemaDrawerVisible}
        onClose={() => setSchemaDrawerVisible(false)}
      >
        <Input.Search
          placeholder="Search tables..."
          style={{ marginBottom: 16 }}
        />
        
        <DirectoryTree
          showLine
          showIcon
          defaultExpandedKeys={['0-0']}
          treeData={tables.map((table, i) => ({
            title: table,
            key: `0-${i}`,
            icon: <TableOutlined />,
            children: tableSchema?.columns.map((col, j) => ({
              title: (
                <Space>
                  {col.name}
                  <Tag color="blue" style={{ fontSize: 10 }}>
                    {col.type}
                  </Tag>
                </Space>
              ),
              key: `0-${i}-${j}`,
              icon: <FileOutlined />,
              isLeaf: true,
            })),
          }))}
          onSelect={(keys, info) => {
            // Handle table selection
            console.log('Selected:', info);
          }}
        />
      </Drawer>
    </div>
  );
};