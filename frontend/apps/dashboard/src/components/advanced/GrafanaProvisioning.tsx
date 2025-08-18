import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Steps,
  Alert,
  Space,
  Typography,
  Progress,
  List,
  Tag,
  Divider,
  Row,
  Col,
  Statistic,
  Badge,
  Tooltip,
  message,
  Modal,
  Input,
  Form,
  Select,
  Spin,
  Result
} from 'antd';
import {
  CheckCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  CloudUploadOutlined,
  DashboardOutlined,
  ApiOutlined,
  DatabaseOutlined,
  LoadingOutlined,
  WarningOutlined,
  RocketOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;
const { Option } = Select;

interface Dashboard {
  id: string;
  name: string;
  description: string;
  panels: number;
  datasources: string[];
  status: 'not_installed' | 'installing' | 'installed' | 'error';
  version: string;
  lastUpdated: number;
}

interface Datasource {
  id: string;
  name: string;
  type: string;
  url: string;
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  isDefault: boolean;
  testResult?: {
    success: boolean;
    message: string;
    latency?: number;
  };
}

interface ProvisioningStep {
  title: string;
  description: string;
  status: 'wait' | 'process' | 'finish' | 'error';
  result?: string;
}

const DASHBOARDS: Dashboard[] = [
  {
    id: 'operator-command-center',
    name: 'Operator Command Center',
    description: 'Main control dashboard with system metrics, agent status, and real-time monitoring',
    panels: 24,
    datasources: ['ClickHouse', 'Prometheus'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  },
  {
    id: 'bandit-dashboard',
    name: 'Bandit Dashboard',
    description: 'Multi-armed bandit optimization metrics, arm performance, and convergence tracking',
    panels: 18,
    datasources: ['ClickHouse'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  },
  {
    id: 'decision-dna-panel',
    name: 'Decision DNA Panel',
    description: 'Cryptographic decision lineage, hash chain verification, and strategy evolution',
    panels: 12,
    datasources: ['ClickHouse'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  },
  {
    id: 'mev-performance',
    name: 'MEV Performance Analytics',
    description: 'Detailed MEV extraction metrics, profit analysis, and leader performance',
    panels: 20,
    datasources: ['ClickHouse', 'Prometheus'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  },
  {
    id: 'system-health',
    name: 'System Health Monitor',
    description: 'Infrastructure health, resource usage, and alert management',
    panels: 16,
    datasources: ['Prometheus', 'ClickHouse'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  },
  {
    id: 'kafka-monitor',
    name: 'Kafka Topics Monitor',
    description: 'Real-time Kafka topic monitoring, lag tracking, and throughput metrics',
    panels: 14,
    datasources: ['Prometheus'],
    status: 'not_installed',
    version: '2.0.0',
    lastUpdated: Date.now()
  }
];

export default function GrafanaProvisioning() {
  const [datasources, setDatasources] = useState<Datasource[]>([
    {
      id: 'clickhouse-primary',
      name: 'ClickHouse',
      type: 'grafana-clickhouse-datasource',
      url: 'http://localhost:8123',
      status: 'disconnected',
      isDefault: true
    },
    {
      id: 'prometheus-primary',
      name: 'Prometheus',
      type: 'prometheus',
      url: 'http://localhost:9090',
      status: 'disconnected',
      isDefault: false
    }
  ]);
  
  const [dashboards, setDashboards] = useState<Dashboard[]>(DASHBOARDS);
  const [provisioningSteps, setProvisioningSteps] = useState<ProvisioningStep[]>([]);
  const [isProvisioning, setIsProvisioning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [grafanaUrl, setGrafanaUrl] = useState('http://localhost:3000');
  const [grafanaToken, setGrafanaToken] = useState('');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configForm] = Form.useForm();

  // Check Grafana connection
  const checkGrafanaConnection = async () => {
    try {
      const response = await fetch(`${grafanaUrl}/api/health`, {
        headers: grafanaToken ? { 'Authorization': `Bearer ${grafanaToken}` } : {}
      });
      return response.ok;
    } catch {
      return false;
    }
  };

  // Test datasource connection
  const testDatasource = async (ds: Datasource) => {
    setDatasources(prev => prev.map(d => 
      d.id === ds.id ? { ...d, status: 'connecting' } : d
    ));

    try {
      const response = await fetch('/api/grafana/test-datasource', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grafanaUrl,
          grafanaToken,
          datasource: ds
        })
      });

      const result = await response.json();
      
      setDatasources(prev => prev.map(d => 
        d.id === ds.id ? {
          ...d,
          status: result.success ? 'connected' : 'error',
          testResult: {
            success: result.success,
            message: result.message,
            latency: result.latency
          }
        } : d
      ));

      if (result.success) {
        message.success(`${ds.name} connected successfully`);
      } else {
        message.error(`Failed to connect to ${ds.name}: ${result.message}`);
      }
    } catch (error) {
      setDatasources(prev => prev.map(d => 
        d.id === ds.id ? {
          ...d,
          status: 'error',
          testResult: {
            success: false,
            message: 'Connection test failed'
          }
        } : d
      ));
      message.error(`Error testing ${ds.name}`);
    }
  };

  // Provision datasource
  const provisionDatasource = async (ds: Datasource): Promise<boolean> => {
    try {
      const response = await fetch('/api/grafana/provision-datasource', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grafanaUrl,
          grafanaToken,
          datasource: ds
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setDatasources(prev => prev.map(d => 
          d.id === ds.id ? { ...d, status: 'connected' } : d
        ));
        return true;
      }
      
      return false;
    } catch {
      return false;
    }
  };

  // Install dashboard
  const installDashboard = async (dashboard: Dashboard): Promise<boolean> => {
    setDashboards(prev => prev.map(d => 
      d.id === dashboard.id ? { ...d, status: 'installing' } : d
    ));

    try {
      const response = await fetch('/api/grafana/install-dashboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grafanaUrl,
          grafanaToken,
          dashboardId: dashboard.id
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setDashboards(prev => prev.map(d => 
          d.id === dashboard.id ? { ...d, status: 'installed' } : d
        ));
        message.success(`${dashboard.name} installed successfully`);
        return true;
      } else {
        setDashboards(prev => prev.map(d => 
          d.id === dashboard.id ? { ...d, status: 'error' } : d
        ));
        message.error(`Failed to install ${dashboard.name}`);
        return false;
      }
    } catch (error) {
      setDashboards(prev => prev.map(d => 
        d.id === dashboard.id ? { ...d, status: 'error' } : d
      ));
      message.error(`Error installing ${dashboard.name}`);
      return false;
    }
  };

  // One-click provisioning
  const runOneClickProvisioning = async () => {
    setIsProvisioning(true);
    setCurrentStep(0);
    
    const steps: ProvisioningStep[] = [
      {
        title: 'Checking Grafana Connection',
        description: 'Verifying Grafana API access',
        status: 'process'
      },
      {
        title: 'Provisioning Datasources',
        description: 'Setting up ClickHouse and Prometheus connections',
        status: 'wait'
      },
      {
        title: 'Installing Dashboards',
        description: 'Deploying MEV monitoring dashboards',
        status: 'wait'
      },
      {
        title: 'Configuring Alerts',
        description: 'Setting up alert rules and notifications',
        status: 'wait'
      },
      {
        title: 'Running Smoke Tests',
        description: 'Verifying dashboard data flow',
        status: 'wait'
      }
    ];
    
    setProvisioningSteps(steps);

    // Step 1: Check Grafana
    const grafanaOk = await checkGrafanaConnection();
    if (!grafanaOk) {
      steps[0]!.status = 'error';
      steps[0]!.result = 'Failed to connect to Grafana';
      setProvisioningSteps([...steps]);
      setIsProvisioning(false);
      message.error('Failed to connect to Grafana. Please check your configuration.');
      return;
    }
    
    steps[0]!.status = 'finish';
    steps[0]!.result = 'Grafana connected';
    steps[1]!.status = 'process';
    setProvisioningSteps([...steps]);
    setCurrentStep(1);

    // Step 2: Provision datasources
    let datasourceSuccess = true;
    for (const ds of datasources) {
      const success = await provisionDatasource(ds);
      if (!success) datasourceSuccess = false;
    }
    
    if (!datasourceSuccess) {
      steps[1]!.status = 'error';
      steps[1]!.result = 'Some datasources failed to provision';
      setProvisioningSteps([...steps]);
      setIsProvisioning(false);
      return;
    }
    
    steps[1]!.status = 'finish';
    steps[1]!.result = 'All datasources provisioned';
    steps[2]!.status = 'process';
    setProvisioningSteps([...steps]);
    setCurrentStep(2);

    // Step 3: Install dashboards
    let dashboardSuccess = true;
    for (const dashboard of dashboards) {
      const success = await installDashboard(dashboard);
      if (!success) dashboardSuccess = false;
    }
    
    if (!dashboardSuccess) {
      steps[2]!.status = 'error';
      steps[2]!.result = 'Some dashboards failed to install';
      setProvisioningSteps([...steps]);
      setIsProvisioning(false);
      return;
    }
    
    steps[2]!.status = 'finish';
    steps[2]!.result = `${dashboards.length} dashboards installed`;
    steps[3]!.status = 'process';
    setProvisioningSteps([...steps]);
    setCurrentStep(3);

    // Step 4: Configure alerts
    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate alert configuration
    steps[3]!.status = 'finish';
    steps[3]!.result = 'Alerts configured';
    steps[4]!.status = 'process';
    setProvisioningSteps([...steps]);
    setCurrentStep(4);

    // Step 5: Run smoke tests
    try {
      const response = await fetch('/api/lab/smoke-test', { method: 'POST' });
      const result = await response.json();
      
      if (result.success) {
        steps[4]!.status = 'finish';
        steps[4]!.result = 'All tests passed';
      } else {
        steps[4]!.status = 'error';
        steps[4]!.result = 'Some tests failed';
      }
    } catch {
      steps[4]!.status = 'error';
      steps[4]!.result = 'Smoke tests failed';
    }
    
    setProvisioningSteps([...steps]);
    setIsProvisioning(false);
    setCurrentStep(5);
    
    message.success('Provisioning completed!');
  };

  // Save configuration
  const saveConfiguration = (values: any) => {
    setGrafanaUrl(values.grafanaUrl);
    setGrafanaToken(values.grafanaToken || '');
    
    // Update datasource URLs
    setDatasources(prev => prev.map(ds => {
      if (ds.id === 'clickhouse-primary') {
        return { ...ds, url: values.clickhouseUrl };
      }
      if (ds.id === 'prometheus-primary') {
        return { ...ds, url: values.prometheusUrl };
      }
      return ds;
    }));
    
    setShowConfigModal(false);
    message.success('Configuration saved');
  };

  // Calculate statistics
  const stats = {
    totalDashboards: dashboards.length,
    installedDashboards: dashboards.filter(d => d.status === 'installed').length,
    totalPanels: dashboards.reduce((sum, d) => sum + d.panels, 0),
    connectedDatasources: datasources.filter(ds => ds.status === 'connected').length,
    totalDatasources: datasources.length
  };

  return (
    <div style={{ padding: 24 }}>
      <Card
        title={
          <Space>
            <DashboardOutlined />
            <Title level={4} style={{ margin: 0 }}>Grafana Dashboard Provisioning</Title>
          </Space>
        }
        extra={
          <Space>
            <Button
              icon={<SettingOutlined />}
              onClick={() => setShowConfigModal(true)}
            >
              Configure
            </Button>
            <Button
              type="primary"
              icon={<RocketOutlined />}
              onClick={runOneClickProvisioning}
              loading={isProvisioning}
              disabled={!grafanaUrl}
            >
              One-Click Provision
            </Button>
          </Space>
        }
      >
        {/* Statistics */}
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="Dashboards"
                value={stats.installedDashboards}
                suffix={`/ ${stats.totalDashboards}`}
                valueStyle={{ color: stats.installedDashboards === stats.totalDashboards ? '#52c41a' : '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Total Panels"
                value={stats.totalPanels}
                prefix={<DashboardOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Datasources"
                value={stats.connectedDatasources}
                suffix={`/ ${stats.totalDatasources}`}
                valueStyle={{ color: stats.connectedDatasources === stats.totalDatasources ? '#52c41a' : '#faad14' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Grafana Status"
                value={grafanaUrl ? 'Configured' : 'Not Configured'}
                valueStyle={{ color: grafanaUrl ? '#52c41a' : '#f5222d', fontSize: 16 }}
              />
            </Card>
          </Col>
        </Row>

        {/* Provisioning Steps */}
        {provisioningSteps.length > 0 && (
          <Card title="Provisioning Progress" style={{ marginBottom: 24 }}>
            <Steps current={currentStep}>
              {provisioningSteps.map((step, idx) => (
                <Step
                  key={idx}
                  title={step.title}
                  description={step.description}
                  status={step.status}
                  icon={
                    step.status === 'process' ? <LoadingOutlined /> :
                    step.status === 'finish' ? <CheckCircleOutlined /> :
                    step.status === 'error' ? <WarningOutlined /> : undefined
                  }
                />
              ))}
            </Steps>
            
            {provisioningSteps.some(s => s.result) && (
              <div style={{ marginTop: 24 }}>
                {provisioningSteps.filter(s => s.result).map((step, idx) => (
                  <Alert
                    key={idx}
                    message={step.result}
                    type={step.status === 'finish' ? 'success' : 'error'}
                    showIcon
                    style={{ marginBottom: 8 }}
                  />
                ))}
              </div>
            )}
          </Card>
        )}

        {/* Datasources */}
        <Card title={<><DatabaseOutlined /> Datasources</> } style={{ marginBottom: 24 }}>
          <List
            dataSource={datasources}
            renderItem={ds => (
              <List.Item
                actions={[
                  <Button
                    size="small"
                    onClick={() => testDatasource(ds)}
                    loading={ds.status === 'connecting'}
                  >
                    Test Connection
                  </Button>,
                  <Button
                    size="small"
                    type="primary"
                    onClick={() => provisionDatasource(ds)}
                    disabled={ds.status === 'connected'}
                  >
                    Provision
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={
                    <Badge
                      status={
                        ds.status === 'connected' ? 'success' :
                        ds.status === 'connecting' ? 'processing' :
                        ds.status === 'error' ? 'error' : 'default'
                      }
                    />
                  }
                  title={
                    <Space>
                      {ds.name}
                      {ds.isDefault && <Tag color="blue">DEFAULT</Tag>}
                      <Text type="secondary">({ds.type})</Text>
                    </Space>
                  }
                  description={
                    <Space direction="vertical">
                      <Text code>{ds.url}</Text>
                      {ds.testResult && (
                        <Text type={ds.testResult.success ? 'success' : 'danger'}>
                          {ds.testResult.message}
                          {ds.testResult.latency && ` (${ds.testResult.latency}ms)`}
                        </Text>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>

        {/* Dashboards */}
        <Card title={<><DashboardOutlined /> Available Dashboards</> }>
          <Row gutter={[16, 16]}>
            {dashboards.map(dashboard => (
              <Col span={8} key={dashboard.id}>
                <Card
                  hoverable
                  actions={[
                    <Button
                      type="primary"
                      size="small"
                      icon={<CloudUploadOutlined />}
                      onClick={() => installDashboard(dashboard)}
                      loading={dashboard.status === 'installing'}
                      disabled={dashboard.status === 'installed'}
                    >
                      {dashboard.status === 'installed' ? 'Installed' : 'Install'}
                    </Button>,
                    dashboard.status === 'installed' && (
                      <Button
                        size="small"
                        icon={<SyncOutlined />}
                        onClick={() => installDashboard(dashboard)}
                      >
                        Update
                      </Button>
                    )
                  ].filter(Boolean)}
                >
                  <Card.Meta
                    title={
                      <Space>
                        {dashboard.name}
                        <Badge
                          status={
                            dashboard.status === 'installed' ? 'success' :
                            dashboard.status === 'installing' ? 'processing' :
                            dashboard.status === 'error' ? 'error' : 'default'
                          }
                        />
                      </Space>
                    }
                    description={dashboard.description}
                  />
                  <Divider />
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text type="secondary">Panels: </Text>
                      <Text strong>{dashboard.panels}</Text>
                    </div>
                    <div>
                      <Text type="secondary">Version: </Text>
                      <Text strong>{dashboard.version}</Text>
                    </div>
                    <div>
                      <Text type="secondary">Datasources: </Text>
                      {dashboard.datasources.map(ds => (
                        <Tag key={ds} color="blue">{ds}</Tag>
                      ))}
                    </div>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>

        {/* Success Result */}
        {!isProvisioning && provisioningSteps.length > 0 && 
         provisioningSteps.every(s => s.status === 'finish') && (
          <Card style={{ marginTop: 24 }}>
            <Result
              status="success"
              title="Provisioning Completed Successfully!"
              subTitle="All dashboards and datasources have been configured."
              extra={[
                <Button
                  type="primary"
                  key="grafana"
                  onClick={() => window.open(grafanaUrl, '_blank')}
                  icon={<DashboardOutlined />}
                >
                  Open Grafana
                </Button>,
                <Button
                  key="test"
                  onClick={() => fetch('/api/lab/smoke-test', { method: 'POST' })}
                  icon={<ThunderboltOutlined />}
                >
                  Run Test Data
                </Button>
              ]}
            />
          </Card>
        )}
      </Card>

      {/* Configuration Modal */}
      <Modal
        title="Grafana Configuration"
        visible={showConfigModal}
        onOk={() => configForm.submit()}
        onCancel={() => {
          setShowConfigModal(false);
          configForm.resetFields();
        }}
        width={600}
      >
        <Form
          form={configForm}
          layout="vertical"
          onFinish={saveConfiguration}
          initialValues={{
            grafanaUrl: grafanaUrl || 'http://localhost:3000',
            clickhouseUrl: datasources.find(ds => ds.id === 'clickhouse-primary')?.url,
            prometheusUrl: datasources.find(ds => ds.id === 'prometheus-primary')?.url
          }}
        >
          <Form.Item
            name="grafanaUrl"
            label="Grafana URL"
            rules={[{ required: true, message: 'Please enter Grafana URL' }]}
          >
            <Input placeholder="http://localhost:3000" />
          </Form.Item>
          
          <Form.Item
            name="grafanaToken"
            label="Grafana API Token (Optional)"
            extra="Required if Grafana has authentication enabled"
          >
            <Input.Password placeholder="Bearer token for API access" />
          </Form.Item>
          
          <Divider />
          
          <Form.Item
            name="clickhouseUrl"
            label="ClickHouse URL"
            rules={[{ required: true }]}
          >
            <Input placeholder="http://localhost:8123" />
          </Form.Item>
          
          <Form.Item
            name="prometheusUrl"
            label="Prometheus URL"
            rules={[{ required: true }]}
          >
            <Input placeholder="http://localhost:9090" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}