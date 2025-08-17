/**
 * ML Model Management Interface
 * Deploy, monitor, and manage machine learning models in production
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Progress,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Upload,
  Alert,
  Statistic,
  Tabs,
  Timeline,
  Badge,
  Tooltip,
  Switch,
  InputNumber,
  DatePicker,
  message,
  notification,
  Drawer,
  Descriptions,
  Typography,
  Divider,
  Skeleton,
  Empty,
  Result,
} from 'antd';
import {
  CloudUploadOutlined,
  RocketOutlined,
  SyncOutlined,
  DeleteOutlined,
  SettingOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  DownloadOutlined,
  HistoryOutlined,
  FundOutlined,
  FileTextOutlined,
  CodeOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import { Line, Area, Column, Gauge, Liquid, Radar } from '@ant-design/plots';
import { apiService, MLModelStatus, ModelSwap } from '../../services/api-service';
import { useMEVStore } from '../../stores/mev-store';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { Dragger } = Upload;
const { RangePicker } = DatePicker;

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  loss: number;
  trainingTime: number;
  inferenceLatency: number;
  throughput: number;
  memoryUsage: number;
  cpuUsage: number;
  gpuUsage?: number;
}

interface ModelVersion {
  version: string;
  timestamp: number;
  accuracy: number;
  status: 'active' | 'inactive' | 'archived';
  deployedBy: string;
  notes: string;
  rollbackAvailable: boolean;
}

interface TrainingJob {
  id: string;
  modelName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: number;
  estimatedCompletion?: number;
  epochs: number;
  currentEpoch: number;
  loss: number;
  learningRate: number;
  batchSize: number;
  datasetSize: number;
}

export const MLModelManager: React.FC = () => {
  const [models, setModels] = useState<MLModelStatus[]>([]);
  const [selectedModel, setSelectedModel] = useState<MLModelStatus | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [trainModalVisible, setTrainModalVisible] = useState(false);
  const [metricsDrawerVisible, setMetricsDrawerVisible] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  const [deployForm] = Form.useForm();
  const [trainForm] = Form.useForm();

  // Fetch model status
  const fetchModels = useCallback(async () => {
    try {
      const modelList = await apiService.getModelStatus();
      setModels(modelList);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  }, []);

  // Fetch model metrics
  const fetchModelMetrics = useCallback(async (modelId: string) => {
    try {
      setLoading(true);
      const metrics = await apiService.getModelMetrics(modelId);
      setModelMetrics(metrics);
    } catch (error) {
      console.error('Failed to fetch model metrics:', error);
      message.error('Failed to load model metrics');
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh
  useEffect(() => {
    fetchModels();
    
    if (autoRefresh) {
      const interval = setInterval(fetchModels, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, fetchModels]);

  // Deploy model
  const handleDeploy = async (values: any) => {
    try {
      setLoading(true);
      
      const modelSwap: ModelSwap = {
        model_id: values.modelId,
        model_path: values.modelPath,
        model_type: values.modelType,
        version: values.version,
        metadata: {
          description: values.description,
          framework: values.framework,
          tags: values.tags?.join(',') || '',
        },
      };

      await apiService.deployModel(modelSwap);
      
      notification.success({
        message: 'Model Deployed',
        description: `Successfully deployed ${values.modelId} v${values.version}`,
        icon: <RocketOutlined style={{ color: '#52c41a' }} />,
      });

      setDeployModalVisible(false);
      deployForm.resetFields();
      fetchModels();
    } catch (error) {
      console.error('Failed to deploy model:', error);
      notification.error({
        message: 'Deployment Failed',
        description: 'Failed to deploy the model. Please check the configuration.',
      });
    } finally {
      setLoading(false);
    }
  };

  // Train model
  const handleTrain = async (values: any) => {
    try {
      setLoading(true);
      
      const trainingConfig = {
        model_name: values.modelName,
        dataset_path: values.datasetPath,
        epochs: values.epochs,
        batch_size: values.batchSize,
        learning_rate: values.learningRate,
        optimizer: values.optimizer,
        loss_function: values.lossFunction,
        validation_split: values.validationSplit,
        early_stopping: values.earlyStopping,
        callbacks: values.callbacks || [],
      };

      await apiService.trainModel(trainingConfig);
      
      notification.success({
        message: 'Training Started',
        description: `Training job for ${values.modelName} has been queued`,
        icon: <ExperimentOutlined style={{ color: '#1890ff' }} />,
      });

      setTrainModalVisible(false);
      trainForm.resetFields();
    } catch (error) {
      console.error('Failed to start training:', error);
      notification.error({
        message: 'Training Failed',
        description: 'Failed to start training job. Please check the configuration.',
      });
    } finally {
      setLoading(false);
    }
  };

  // Rollback model
  const handleRollback = async (modelId: string, version: string) => {
    Modal.confirm({
      title: 'Rollback Model',
      content: `Are you sure you want to rollback ${modelId} to version ${version}?`,
      icon: <WarningOutlined />,
      onOk: async () => {
        try {
          // Implement rollback logic
          message.success(`Rolled back ${modelId} to version ${version}`);
          fetchModels();
        } catch (error) {
          message.error('Rollback failed');
        }
      },
    });
  };

  // Delete model
  const handleDelete = async (modelId: string) => {
    Modal.confirm({
      title: 'Delete Model',
      content: `Are you sure you want to delete ${modelId}? This action cannot be undone.`,
      icon: <DeleteOutlined style={{ color: '#ff4d4f' }} />,
      okText: 'Delete',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          // Implement delete logic
          message.success(`Deleted model ${modelId}`);
          fetchModels();
        } catch (error) {
          message.error('Failed to delete model');
        }
      },
    });
  };

  // Model status indicator
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready':
        return 'success';
      case 'loading':
        return 'processing';
      case 'training':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  // Performance gauge config
  const gaugeConfig = useMemo(() => ({
    percent: selectedModel ? selectedModel.accuracy : 0,
    range: {
      color: ['#30BF78', '#FAAD14', '#F4664A'],
      width: 12,
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    axis: {
      label: {
        formatter: (v: string) => `${Number(v) * 100}%`,
      },
      subTickLine: {
        count: 3,
      },
    },
    statistic: {
      title: {
        formatter: () => 'Accuracy',
        style: {
          fontSize: '14px',
          color: '#8c8c8c',
        },
      },
      content: {
        formatter: (datum: any) => `${(datum.percent * 100).toFixed(1)}%`,
        style: {
          fontSize: '24px',
          color: datum?.percent > 0.9 ? '#30BF78' : '#666',
        },
      },
    },
  }), [selectedModel]);

  // Model table columns
  const modelColumns = [
    {
      title: 'Model ID',
      dataIndex: 'model_id',
      key: 'model_id',
      render: (text: string) => (
        <Space>
          <ExperimentOutlined />
          <Text strong>{text}</Text>
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge status={getStatusColor(status) as any} text={status.toUpperCase()} />
      ),
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => (
        <Progress
          percent={accuracy * 100}
          size="small"
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
      ),
    },
    {
      title: 'Predictions/sec',
      dataIndex: 'predictions_per_sec',
      key: 'predictions_per_sec',
      render: (value: number) => (
        <Statistic
          value={value}
          suffix="/s"
          valueStyle={{ fontSize: '14px' }}
        />
      ),
    },
    {
      title: 'Memory',
      dataIndex: 'memory_usage_mb',
      key: 'memory_usage_mb',
      render: (mb: number) => `${mb.toFixed(0)} MB`,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: MLModelStatus) => (
        <Space>
          <Tooltip title="View Metrics">
            <Button
              icon={<LineChartOutlined />}
              size="small"
              onClick={() => {
                setSelectedModel(record);
                fetchModelMetrics(record.model_id);
                setMetricsDrawerVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Settings">
            <Button icon={<SettingOutlined />} size="small" />
          </Tooltip>
          <Tooltip title="Delete">
            <Button
              icon={<DeleteOutlined />}
              size="small"
              danger
              onClick={() => handleDelete(record.model_id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // Training jobs table columns
  const trainingColumns = [
    {
      title: 'Job ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
    },
    {
      title: 'Model',
      dataIndex: 'modelName',
      key: 'modelName',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors: Record<string, string> = {
          pending: 'default',
          running: 'processing',
          completed: 'success',
          failed: 'error',
        };
        return <Tag color={colors[status]}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_: any, record: TrainingJob) => (
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'running' ? 'active' : undefined}
          />
          <Text type="secondary">
            Epoch {record.currentEpoch}/{record.epochs}
          </Text>
        </Space>
      ),
    },
    {
      title: 'Loss',
      dataIndex: 'loss',
      key: 'loss',
      render: (loss: number) => loss.toFixed(4),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: TrainingJob) => (
        <Space>
          {record.status === 'running' && (
            <Button size="small" danger>
              Stop
            </Button>
          )}
          <Button size="small">View Logs</Button>
        </Space>
      ),
    },
  ];

  return (
    <div className="ml-model-manager">
      {/* Header */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Row align="middle" justify="space-between">
              <Col>
                <Space>
                  <ExperimentOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                  <div>
                    <Title level={4} style={{ margin: 0 }}>
                      ML Model Management
                    </Title>
                    <Text type="secondary">
                      Deploy, monitor, and manage machine learning models
                    </Text>
                  </div>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Switch
                    checkedChildren="Auto Refresh"
                    unCheckedChildren="Manual"
                    checked={autoRefresh}
                    onChange={setAutoRefresh}
                  />
                  <Button
                    icon={<CloudUploadOutlined />}
                    type="primary"
                    onClick={() => setDeployModalVisible(true)}
                  >
                    Deploy Model
                  </Button>
                  <Button
                    icon={<ExperimentOutlined />}
                    onClick={() => setTrainModalVisible(true)}
                  >
                    Train Model
                  </Button>
                  <Button icon={<SyncOutlined spin={loading} />} onClick={fetchModels}>
                    Refresh
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Statistics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Models"
              value={models.length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active Models"
              value={models.filter(m => m.status === 'ready').length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Training Jobs"
              value={trainingJobs.filter(j => j.status === 'running').length}
              prefix={<LoadingOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Avg Accuracy"
              value={
                models.length > 0
                  ? (models.reduce((sum, m) => sum + m.accuracy, 0) / models.length * 100).toFixed(1)
                  : 0
              }
              suffix="%"
              prefix={<FundOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Main Content */}
      <Tabs defaultActiveKey="models">
        <TabPane tab={<span><DatabaseOutlined /> Deployed Models</span>} key="models">
          <Card>
            <Table
              columns={modelColumns}
              dataSource={models}
              rowKey="model_id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `Total ${total} models`,
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><ExperimentOutlined /> Training Jobs</span>} key="training">
          <Card>
            {trainingJobs.length > 0 ? (
              <Table
                columns={trainingColumns}
                dataSource={trainingJobs}
                rowKey="id"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                }}
              />
            ) : (
              <Empty
                description="No training jobs"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={() => setTrainModalVisible(true)}
                >
                  Start Training
                </Button>
              </Empty>
            )}
          </Card>
        </TabPane>

        <TabPane tab={<span><HistoryOutlined /> Version History</span>} key="versions">
          <Card>
            <Timeline mode="left">
              {modelVersions.map((version) => (
                <Timeline.Item
                  key={version.version}
                  color={version.status === 'active' ? 'green' : 'gray'}
                  label={new Date(version.timestamp).toLocaleString()}
                >
                  <Card size="small">
                    <Space direction="vertical" size="small">
                      <Space>
                        <Tag color={version.status === 'active' ? 'success' : 'default'}>
                          v{version.version}
                        </Tag>
                        <Text strong>Accuracy: {(version.accuracy * 100).toFixed(1)}%</Text>
                      </Space>
                      <Text type="secondary">Deployed by {version.deployedBy}</Text>
                      <Paragraph ellipsis={{ rows: 2 }}>
                        {version.notes}
                      </Paragraph>
                      {version.rollbackAvailable && (
                        <Button
                          size="small"
                          onClick={() => handleRollback('model-id', version.version)}
                        >
                          Rollback to this version
                        </Button>
                      )}
                    </Space>
                  </Card>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* Deploy Model Modal */}
      <Modal
        title={<Space><RocketOutlined /> Deploy New Model</Space>}
        visible={deployModalVisible}
        onCancel={() => setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={deployForm}
          layout="vertical"
          onFinish={handleDeploy}
        >
          <Form.Item
            name="modelId"
            label="Model ID"
            rules={[{ required: true, message: 'Please enter model ID' }]}
          >
            <Input placeholder="e.g., mev-predictor-v3" />
          </Form.Item>

          <Form.Item
            name="modelPath"
            label="Model Path"
            rules={[{ required: true, message: 'Please enter model path' }]}
          >
            <Input placeholder="e.g., s3://models/mev-predictor-v3.pkl" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="modelType"
                label="Model Type"
                rules={[{ required: true }]}
              >
                <Select placeholder="Select model type">
                  <Option value="xgboost">XGBoost</Option>
                  <Option value="lightgbm">LightGBM</Option>
                  <Option value="tensorflow">TensorFlow</Option>
                  <Option value="pytorch">PyTorch</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="version"
                label="Version"
                rules={[{ required: true }]}
              >
                <Input placeholder="e.g., 1.0.0" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="framework"
            label="Framework Version"
          >
            <Input placeholder="e.g., tensorflow==2.10.0" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <Input.TextArea rows={3} placeholder="Model description..." />
          </Form.Item>

          <Form.Item
            name="tags"
            label="Tags"
          >
            <Select mode="tags" placeholder="Add tags...">
              <Option value="production">Production</Option>
              <Option value="staging">Staging</Option>
              <Option value="experimental">Experimental</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Deploy
              </Button>
              <Button onClick={() => setDeployModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Train Model Modal */}
      <Modal
        title={<Space><ExperimentOutlined /> Train New Model</Space>}
        visible={trainModalVisible}
        onCancel={() => setTrainModalVisible(false)}
        footer={null}
        width={700}
      >
        <Form
          form={trainForm}
          layout="vertical"
          onFinish={handleTrain}
        >
          <Form.Item
            name="modelName"
            label="Model Name"
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., mev-predictor" />
          </Form.Item>

          <Form.Item
            name="datasetPath"
            label="Dataset Path"
            rules={[{ required: true }]}
          >
            <Input placeholder="e.g., s3://datasets/mev-training-data.parquet" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="epochs"
                label="Epochs"
                rules={[{ required: true }]}
                initialValue={100}
              >
                <InputNumber min={1} max={1000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="batchSize"
                label="Batch Size"
                rules={[{ required: true }]}
                initialValue={32}
              >
                <InputNumber min={1} max={512} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="learningRate"
                label="Learning Rate"
                rules={[{ required: true }]}
                initialValue={0.001}
              >
                <InputNumber
                  min={0.00001}
                  max={1}
                  step={0.0001}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="optimizer"
                label="Optimizer"
                rules={[{ required: true }]}
                initialValue="adam"
              >
                <Select>
                  <Option value="adam">Adam</Option>
                  <Option value="sgd">SGD</Option>
                  <Option value="rmsprop">RMSprop</Option>
                  <Option value="adagrad">Adagrad</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="lossFunction"
                label="Loss Function"
                rules={[{ required: true }]}
                initialValue="mse"
              >
                <Select>
                  <Option value="mse">MSE</Option>
                  <Option value="mae">MAE</Option>
                  <Option value="binary_crossentropy">Binary Crossentropy</Option>
                  <Option value="categorical_crossentropy">Categorical Crossentropy</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="validationSplit"
            label="Validation Split"
            initialValue={0.2}
          >
            <InputNumber
              min={0.1}
              max={0.5}
              step={0.05}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item
            name="earlyStopping"
            label="Early Stopping"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch checkedChildren="Enabled" unCheckedChildren="Disabled" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Start Training
              </Button>
              <Button onClick={() => setTrainModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Metrics Drawer */}
      <Drawer
        title={
          <Space>
            <LineChartOutlined />
            Model Metrics: {selectedModel?.model_id}
          </Space>
        }
        placement="right"
        width={720}
        visible={metricsDrawerVisible}
        onClose={() => setMetricsDrawerVisible(false)}
      >
        {selectedModel && modelMetrics && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {/* Performance Gauge */}
            <Card title="Model Performance">
              <Row gutter={16}>
                <Col span={12}>
                  <Gauge {...gaugeConfig} />
                </Col>
                <Col span={12}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Statistic
                      title="F1 Score"
                      value={modelMetrics.f1Score}
                      precision={3}
                      valueStyle={{ color: '#3f8600' }}
                    />
                    <Statistic
                      title="AUC"
                      value={modelMetrics.auc}
                      precision={3}
                      valueStyle={{ color: '#cf1322' }}
                    />
                    <Statistic
                      title="Inference Latency"
                      value={modelMetrics.inferenceLatency}
                      suffix="ms"
                      precision={2}
                    />
                  </Space>
                </Col>
              </Row>
            </Card>

            {/* Detailed Metrics */}
            <Card title="Detailed Metrics">
              <Descriptions column={2} bordered size="small">
                <Descriptions.Item label="Accuracy">
                  {(modelMetrics.accuracy * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label="Precision">
                  {(modelMetrics.precision * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label="Recall">
                  {(modelMetrics.recall * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label="Loss">
                  {modelMetrics.loss.toFixed(4)}
                </Descriptions.Item>
                <Descriptions.Item label="Training Time">
                  {modelMetrics.trainingTime.toFixed(0)}s
                </Descriptions.Item>
                <Descriptions.Item label="Throughput">
                  {modelMetrics.throughput.toFixed(0)}/s
                </Descriptions.Item>
                <Descriptions.Item label="Memory Usage">
                  {modelMetrics.memoryUsage.toFixed(0)} MB
                </Descriptions.Item>
                <Descriptions.Item label="CPU Usage">
                  {modelMetrics.cpuUsage.toFixed(1)}%
                </Descriptions.Item>
                {modelMetrics.gpuUsage !== undefined && (
                  <Descriptions.Item label="GPU Usage">
                    {modelMetrics.gpuUsage.toFixed(1)}%
                  </Descriptions.Item>
                )}
              </Descriptions>
            </Card>

            {/* Actions */}
            <Card title="Actions">
              <Space wrap>
                <Button icon={<DownloadOutlined />}>
                  Download Model
                </Button>
                <Button icon={<CodeOutlined />}>
                  View Code
                </Button>
                <Button icon={<FileTextOutlined />}>
                  Export Report
                </Button>
                <Button icon={<BranchesOutlined />}>
                  Create Version
                </Button>
                <Button danger icon={<DeleteOutlined />}>
                  Delete Model
                </Button>
              </Space>
            </Card>
          </Space>
        )}
      </Drawer>
    </div>
  );
};