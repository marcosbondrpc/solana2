import React, { useState, useEffect, useCallback } from 'react';
import { 
  Card, 
  Select, 
  Button, 
  Input, 
  Space, 
  Table, 
  Form, 
  Row, 
  Col,
  Tag,
  Tooltip,
  message,
  Modal,
  Tabs,
  Alert,
  Spin,
  Typography,
  Divider,
  Badge,
  Statistic
} from 'antd';
import {
  DatabaseOutlined,
  PlayCircleOutlined,
  SaveOutlined,
  DownloadOutlined,
  PlusOutlined,
  DeleteOutlined,
  CopyOutlined,
  HistoryOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Title, Text, Paragraph } = Typography;

interface QueryCondition {
  id: string;
  column: string;
  operator: string;
  value: string;
  conjunction: 'AND' | 'OR';
}

interface SavedQuery {
  id: string;
  name: string;
  query: string;
  description: string;
  created: number;
  lastRun: number;
  runCount: number;
}

interface QueryResult {
  columns: string[];
  rows: any[][];
  rowCount: number;
  executionTime: number;
  bytesRead: number;
}

const TABLES = {
  'bandit_events': {
    columns: ['ts', 'module', 'policy', 'route', 'arm', 'payoff', 'tip_sol', 'ev_sol', 'slot', 'leader', 'landed', 'p_land_est'],
    description: 'Bandit optimization events'
  },
  'mev_opportunities': {
    columns: ['ts', 'slot', 'leader', 'profit_est_sol', 'tip_sol', 'route', 'landed', 'dna_fp', 'model_id', 'policy'],
    description: 'MEV opportunity tracking with Decision DNA'
  },
  'control_acks': {
    columns: ['ts', 'module', 'cmd_type', 'params', 'result', 'latency_us'],
    description: 'Control plane acknowledgments'
  },
  'bandit_arm_rollup': {
    columns: ['minute', 'arm_id', 'route', 'tip_ladder', 'pulls', 'lands', 'total_payoff', 'avg_payoff', 'land_rate'],
    description: 'Minute-level bandit arm statistics'
  },
  'system_metrics': {
    columns: ['ts', 'cpu_usage', 'memory_usage', 'network_in', 'network_out', 'disk_io', 'latency_p99'],
    description: 'System performance metrics'
  }
};

const OPERATORS = {
  '=': 'equals',
  '!=': 'not equals',
  '>': 'greater than',
  '>=': 'greater or equal',
  '<': 'less than',
  '<=': 'less or equal',
  'LIKE': 'contains',
  'NOT LIKE': 'not contains',
  'IN': 'in list',
  'NOT IN': 'not in list',
  'IS NULL': 'is null',
  'IS NOT NULL': 'is not null'
};

const AGGREGATIONS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT'];
const TIME_FUNCTIONS = ['toStartOfMinute', 'toStartOfHour', 'toStartOfDay', 'toStartOfWeek', 'toStartOfMonth'];

export default function ClickHouseQueryBuilder() {
  const [selectedTable, setSelectedTable] = useState<string>('');
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [conditions, setConditions] = useState<QueryCondition[]>([]);
  const [groupBy, setGroupBy] = useState<string[]>([]);
  const [orderBy, setOrderBy] = useState<{ column: string; direction: 'ASC' | 'DESC' }>({ column: '', direction: 'DESC' });
  const [limit, setLimit] = useState<number>(100);
  const [rawQuery, setRawQuery] = useState<string>('');
  const [queryMode, setQueryMode] = useState<'builder' | 'raw'>('builder');
  const [loading, setLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveForm] = Form.useForm();

  // Load saved queries from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('clickhouse_saved_queries');
    if (saved) {
      setSavedQueries(JSON.parse(saved));
    }
    
    const history = localStorage.getItem('clickhouse_query_history');
    if (history) {
      setQueryHistory(JSON.parse(history));
    }
  }, []);

  // Build SQL query from builder state
  const buildQuery = useCallback(() => {
    if (!selectedTable) return '';
    
    let query = 'SELECT ';
    
    // Columns
    if (selectedColumns.length === 0) {
      query += '*';
    } else {
      query += selectedColumns.join(', ');
    }
    
    // FROM
    query += `\nFROM ${selectedTable}`;
    
    // WHERE
    if (conditions.length > 0) {
      query += '\nWHERE ';
      conditions.forEach((cond, idx) => {
        if (idx > 0) query += ` ${cond.conjunction} `;
        
        if (cond.operator === 'IS NULL' || cond.operator === 'IS NOT NULL') {
          query += `${cond.column} ${cond.operator}`;
        } else if (cond.operator === 'IN' || cond.operator === 'NOT IN') {
          query += `${cond.column} ${cond.operator} (${cond.value})`;
        } else if (cond.operator === 'LIKE' || cond.operator === 'NOT LIKE') {
          query += `${cond.column} ${cond.operator} '%${cond.value}%'`;
        } else {
          query += `${cond.column} ${cond.operator} '${cond.value}'`;
        }
      });
    }
    
    // GROUP BY
    if (groupBy.length > 0) {
      query += `\nGROUP BY ${groupBy.join(', ')}`;
    }
    
    // ORDER BY
    if (orderBy.column) {
      query += `\nORDER BY ${orderBy.column} ${orderBy.direction}`;
    }
    
    // LIMIT
    query += `\nLIMIT ${limit}`;
    
    return query;
  }, [selectedTable, selectedColumns, conditions, groupBy, orderBy, limit]);

  // Update raw query when builder changes
  useEffect(() => {
    if (queryMode === 'builder') {
      setRawQuery(buildQuery());
    }
  }, [queryMode, buildQuery]);

  // Execute query
  const executeQuery = async () => {
    const query = queryMode === 'builder' ? buildQuery() : rawQuery;
    
    if (!query) {
      message.error('Please build or enter a query');
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch('/api/clickhouse/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      
      const result = await response.json();
      
      if (response.ok) {
        setQueryResult(result);
        message.success(`Query executed in ${result.executionTime}ms`);
        
        // Add to history
        const newHistory = [query, ...queryHistory.filter(q => q !== query)].slice(0, 50);
        setQueryHistory(newHistory);
        localStorage.setItem('clickhouse_query_history', JSON.stringify(newHistory));
      } else {
        message.error(result.error || 'Query execution failed');
      }
    } catch (error) {
      message.error('Failed to execute query');
      console.error('Query execution error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Add condition
  const addCondition = () => {
    const newCondition: QueryCondition = {
      id: Date.now().toString(),
      column: '',
      operator: '=',
      value: '',
      conjunction: 'AND'
    };
    setConditions([...conditions, newCondition]);
  };

  // Remove condition
  const removeCondition = (id: string) => {
    setConditions(conditions.filter(c => c.id !== id));
  };

  // Update condition
  const updateCondition = (id: string, field: keyof QueryCondition, value: any) => {
    setConditions(conditions.map(c => 
      c.id === id ? { ...c, [field]: value } : c
    ));
  };

  // Save query
  const saveQuery = async (values: any) => {
    const query = queryMode === 'builder' ? buildQuery() : rawQuery;
    
    const newQuery: SavedQuery = {
      id: Date.now().toString(),
      name: values.name,
      description: values.description || '',
      query,
      created: Date.now(),
      lastRun: 0,
      runCount: 0
    };
    
    const updatedQueries = [...savedQueries, newQuery];
    setSavedQueries(updatedQueries);
    localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updatedQueries));
    
    setShowSaveModal(false);
    saveForm.resetFields();
    message.success('Query saved successfully');
  };

  // Load saved query
  const loadSavedQuery = (query: SavedQuery) => {
    setRawQuery(query.query);
    setQueryMode('raw');
    
    // Update run stats
    const updated = savedQueries.map(q => 
      q.id === query.id 
        ? { ...q, lastRun: Date.now(), runCount: q.runCount + 1 }
        : q
    );
    setSavedQueries(updated);
    localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updated));
  };

  // Delete saved query
  const deleteSavedQuery = (id: string) => {
    const updated = savedQueries.filter(q => q.id !== id);
    setSavedQueries(updated);
    localStorage.setItem('clickhouse_saved_queries', JSON.stringify(updated));
    message.success('Query deleted');
  };

  // Export results
  const exportResults = () => {
    if (!queryResult) return;
    
    const csv = [
      queryResult.columns.join(','),
      ...queryResult.rows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    message.success('Results exported to CSV');
  };

  // Copy query to clipboard
  const copyQuery = () => {
    const query = queryMode === 'builder' ? buildQuery() : rawQuery;
    navigator.clipboard.writeText(query);
    message.success('Query copied to clipboard');
  };

  return (
    <div style={{ padding: 24 }}>
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <Title level={4} style={{ margin: 0 }}>ClickHouse Query Builder</Title>
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={executeQuery}
              loading={loading}
            >
              Execute Query
            </Button>
            <Button
              icon={<SaveOutlined />}
              onClick={() => setShowSaveModal(true)}
            >
              Save Query
            </Button>
            <Button
              icon={<CopyOutlined />}
              onClick={copyQuery}
            >
              Copy SQL
            </Button>
          </Space>
        }
      >
        <Tabs
          activeKey={queryMode}
          onChange={(key) => setQueryMode(key as 'builder' | 'raw')}
        >
          <TabPane tab="Query Builder" key="builder">
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {/* Table Selection */}
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="Table">
                    <Select
                      placeholder="Select a table"
                      value={selectedTable}
                      onChange={(value) => {
                        setSelectedTable(value);
                        setSelectedColumns([]);
                        setConditions([]);
                        setGroupBy([]);
                      }}
                      style={{ width: '100%' }}
                    >
                      {Object.entries(TABLES).map(([table, info]) => (
                        <Option key={table} value={table}>
                          <Space>
                            <DatabaseOutlined />
                            <span>{table}</span>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              - {info.description}
                            </Text>
                          </Space>
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="Columns">
                    <Select
                      mode="multiple"
                      placeholder="Select columns (leave empty for *)"
                      value={selectedColumns}
                      onChange={setSelectedColumns}
                      style={{ width: '100%' }}
                      disabled={!selectedTable}
                    >
                      {selectedTable && TABLES[selectedTable as keyof typeof TABLES]?.columns.map(col => (
                        <Option key={col} value={col}>{col}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

              {/* Conditions */}
              <Card size="small" title="WHERE Conditions">
                {conditions.map((condition, index) => (
                  <Row key={condition.id} gutter={8} style={{ marginBottom: 8 }}>
                    {index > 0 && (
                      <Col span={2}>
                        <Select
                          value={condition.conjunction}
                          onChange={(value) => updateCondition(condition.id, 'conjunction', value)}
                          style={{ width: '100%' }}
                        >
                          <Option value="AND">AND</Option>
                          <Option value="OR">OR</Option>
                        </Select>
                      </Col>
                    )}
                    <Col span={index > 0 ? 6 : 8}>
                      <Select
                        placeholder="Column"
                        value={condition.column}
                        onChange={(value) => updateCondition(condition.id, 'column', value)}
                        style={{ width: '100%' }}
                      >
                        {selectedTable && TABLES[selectedTable as keyof typeof TABLES]?.columns.map(col => (
                          <Option key={col} value={col}>{col}</Option>
                        ))}
                      </Select>
                    </Col>
                    <Col span={6}>
                      <Select
                        value={condition.operator}
                        onChange={(value) => updateCondition(condition.id, 'operator', value)}
                        style={{ width: '100%' }}
                      >
                        {Object.entries(OPERATORS).map(([op, desc]) => (
                          <Option key={op} value={op}>{desc}</Option>
                        ))}
                      </Select>
                    </Col>
                    <Col span={8}>
                      <Input
                        placeholder="Value"
                        value={condition.value}
                        onChange={(e) => updateCondition(condition.id, 'value', e.target.value)}
                        disabled={condition.operator === 'IS NULL' || condition.operator === 'IS NOT NULL'}
                      />
                    </Col>
                    <Col span={2}>
                      <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => removeCondition(condition.id)}
                      />
                    </Col>
                  </Row>
                ))}
                <Button
                  type="dashed"
                  icon={<PlusOutlined />}
                  onClick={addCondition}
                  style={{ width: '100%' }}
                >
                  Add Condition
                </Button>
              </Card>

              {/* GROUP BY and ORDER BY */}
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item label="GROUP BY">
                    <Select
                      mode="multiple"
                      placeholder="Select columns"
                      value={groupBy}
                      onChange={setGroupBy}
                      style={{ width: '100%' }}
                      disabled={!selectedTable}
                    >
                      {selectedColumns.map(col => (
                        <Option key={col} value={col}>{col}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item label="ORDER BY">
                    <Select
                      placeholder="Select column"
                      value={orderBy.column}
                      onChange={(value) => setOrderBy({ ...orderBy, column: value })}
                      style={{ width: '100%' }}
                      disabled={!selectedTable}
                    >
                      {(selectedColumns.length > 0 ? selectedColumns : 
                        TABLES[selectedTable as keyof typeof TABLES]?.columns || []
                      ).map(col => (
                        <Option key={col} value={col}>{col}</Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="Direction">
                    <Select
                      value={orderBy.direction}
                      onChange={(value) => setOrderBy({ ...orderBy, direction: value })}
                      style={{ width: '100%' }}
                    >
                      <Option value="ASC">ASC</Option>
                      <Option value="DESC">DESC</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="LIMIT">
                    <Input
                      type="number"
                      value={limit}
                      onChange={(e) => setLimit(parseInt(e.target.value) || 100)}
                      min={1}
                      max={10000}
                    />
                  </Form.Item>
                </Col>
              </Row>

              {/* Generated SQL */}
              <Card size="small" title="Generated SQL">
                <SyntaxHighlighter
                  language="sql"
                  style={vscDarkPlus}
                  customStyle={{ fontSize: 12, borderRadius: 4 }}
                >
                  {buildQuery() || '-- Select a table to begin'}
                </SyntaxHighlighter>
              </Card>
            </Space>
          </TabPane>

          <TabPane tab="Raw SQL" key="raw">
            <TextArea
              rows={10}
              value={rawQuery}
              onChange={(e) => setRawQuery(e.target.value)}
              placeholder="Enter your SQL query here..."
              style={{ fontFamily: 'monospace', fontSize: 13 }}
            />
            
            {/* Query Templates */}
            <Card size="small" title="Quick Templates" style={{ marginTop: 16 }}>
              <Space wrap>
                <Button
                  size="small"
                  onClick={() => setRawQuery(`SELECT * FROM bandit_events ORDER BY ts DESC LIMIT 100`)}
                >
                  Recent Bandit Events
                </Button>
                <Button
                  size="small"
                  onClick={() => setRawQuery(`SELECT 
  toStartOfMinute(ts) as minute,
  route,
  COUNT(*) as events,
  AVG(payoff) as avg_payoff,
  SUM(landed) / COUNT(*) as land_rate
FROM bandit_events
WHERE ts > now() - INTERVAL 1 HOUR
GROUP BY minute, route
ORDER BY minute DESC`)}
                >
                  Bandit Stats (1h)
                </Button>
                <Button
                  size="small"
                  onClick={() => setRawQuery(`SELECT 
  leader,
  COUNT(*) as opportunities,
  SUM(profit_est_sol) as total_profit,
  AVG(tip_sol) as avg_tip
FROM mev_opportunities
WHERE ts > now() - INTERVAL 24 HOUR
GROUP BY leader
ORDER BY total_profit DESC
LIMIT 20`)}
                >
                  Leader Performance
                </Button>
              </Space>
            </Card>
          </TabPane>
        </Tabs>
      </Card>

      {/* Query Results */}
      {queryResult && (
        <Card
          title={
            <Space>
              <ThunderboltOutlined />
              <span>Query Results</span>
              <Badge count={queryResult.rowCount} style={{ backgroundColor: '#52c41a' }} />
            </Space>
          }
          style={{ marginTop: 24 }}
          extra={
            <Space>
              <Statistic
                title="Execution Time"
                value={queryResult.executionTime}
                suffix="ms"
                valueStyle={{ fontSize: 14 }}
              />
              <Statistic
                title="Bytes Read"
                value={(queryResult.bytesRead / 1024).toFixed(2)}
                suffix="KB"
                valueStyle={{ fontSize: 14 }}
              />
              <Button
                icon={<DownloadOutlined />}
                onClick={exportResults}
              >
                Export CSV
              </Button>
            </Space>
          }
        >
          <Table
            columns={queryResult.columns.map(col => ({
              title: col,
              dataIndex: col,
              key: col,
              ellipsis: true,
              render: (value: any) => {
                if (value === null) return <Tag>NULL</Tag>;
                if (typeof value === 'boolean') return value ? <Tag color="green">true</Tag> : <Tag color="red">false</Tag>;
                if (typeof value === 'number') return <Text code>{value}</Text>;
                return <Tooltip title={value}><Text ellipsis>{value}</Text></Tooltip>;
              }
            }))}
            dataSource={queryResult.rows.map((row, idx) => {
              const obj: any = { key: idx };
              queryResult.columns.forEach((col, colIdx) => {
                obj[col] = row[colIdx];
              });
              return obj;
            })}
            scroll={{ x: true }}
            pagination={{
              pageSize: 50,
              showSizeChanger: true,
              showTotal: (total) => `Total ${total} rows`
            }}
          />
        </Card>
      )}

      {/* Saved Queries */}
      <Card
        title={<><SaveOutlined /> Saved Queries</>}
        style={{ marginTop: 24 }}
      >
        {savedQueries.length === 0 ? (
          <Alert
            message="No saved queries yet"
            description="Execute a query and click 'Save Query' to save it for later use"
            type="info"
          />
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            {savedQueries.map(query => (
              <Card
                key={query.id}
                size="small"
                hoverable
                onClick={() => loadSavedQuery(query)}
                extra={
                  <Space>
                    <Tag>{query.runCount} runs</Tag>
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSavedQuery(query.id);
                      }}
                    />
                  </Space>
                }
              >
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>{query.name}</Text>
                  {query.description && <Text type="secondary">{query.description}</Text>}
                  <Text code style={{ fontSize: 11 }}>
                    {query.query.length > 100 ? query.query.substring(0, 100) + '...' : query.query}
                  </Text>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    Created: {new Date(query.created).toLocaleString()}
                    {query.lastRun > 0 && ` | Last run: ${new Date(query.lastRun).toLocaleString()}`}
                  </Text>
                </Space>
              </Card>
            ))}
          </Space>
        )}
      </Card>

      {/* Query History */}
      <Card
        title={<><HistoryOutlined /> Query History</>}
        style={{ marginTop: 24 }}
        bodyStyle={{ maxHeight: 300, overflowY: 'auto' }}
      >
        {queryHistory.length === 0 ? (
          <Alert
            message="No query history"
            description="Your executed queries will appear here"
            type="info"
          />
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            {queryHistory.slice(0, 10).map((query, idx) => (
              <Card
                key={idx}
                size="small"
                hoverable
                onClick={() => {
                  setRawQuery(query);
                  setQueryMode('raw');
                }}
              >
                <Text code style={{ fontSize: 11 }}>
                  {query.length > 200 ? query.substring(0, 200) + '...' : query}
                </Text>
              </Card>
            ))}
          </Space>
        )}
      </Card>

      {/* Save Query Modal */}
      <Modal
        title="Save Query"
        visible={showSaveModal}
        onOk={() => saveForm.submit()}
        onCancel={() => {
          setShowSaveModal(false);
          saveForm.resetFields();
        }}
      >
        <Form
          form={saveForm}
          layout="vertical"
          onFinish={saveQuery}
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
            <TextArea
              rows={3}
              placeholder="Optional description of what this query does"
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}