import { Router } from 'express';
import axios from 'axios';
import multer from 'multer';

const router = Router();
const upload = multer({ dest: '/tmp/uploads/' });

// Data scrapper service configuration
const SCRAPPER_URL = process.env.SCRAPPER_URL || 'http://localhost:8082';

// Scrapper control endpoints
router.post('/scrapper/start', async (req, res) => {
  try {
    const response = await axios.post(`${SCRAPPER_URL}/api/scrapper/start`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error starting scrapper:', error);
    res.status(500).json({ error: 'Failed to start scrapper' });
  }
});

router.post('/scrapper/stop', async (req, res) => {
  try {
    const response = await axios.post(`${SCRAPPER_URL}/api/scrapper/stop`);
    res.json(response.data);
  } catch (error) {
    console.error('Error stopping scrapper:', error);
    res.status(500).json({ error: 'Failed to stop scrapper' });
  }
});

router.get('/scrapper/status', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/scrapper/status`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching scrapper status:', error);
    res.status(500).json({ error: 'Failed to fetch scrapper status' });
  }
});

router.get('/scrapper/progress', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/scrapper/progress`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching progress:', error);
    res.status(500).json({ error: 'Failed to fetch progress' });
  }
});

// Dataset management endpoints
router.get('/datasets', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/datasets`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error listing datasets:', error);
    res.status(500).json({ error: 'Failed to list datasets' });
  }
});

router.post('/datasets', async (req, res) => {
  try {
    const response = await axios.post(`${SCRAPPER_URL}/api/datasets`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error creating dataset:', error);
    res.status(500).json({ error: 'Failed to create dataset' });
  }
});

router.get('/datasets/:id', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/datasets/${req.params.id}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching dataset:', error);
    res.status(500).json({ error: 'Failed to fetch dataset' });
  }
});

router.delete('/datasets/:id', async (req, res) => {
  try {
    const response = await axios.delete(`${SCRAPPER_URL}/api/datasets/${req.params.id}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting dataset:', error);
    res.status(500).json({ error: 'Failed to delete dataset' });
  }
});

router.post('/datasets/:id/export', async (req, res) => {
  try {
    const response = await axios.post(
      `${SCRAPPER_URL}/api/datasets/${req.params.id}/export`,
      req.body,
      {
        responseType: 'stream'
      }
    );
    
    // Set appropriate headers for file download
    res.setHeader('Content-Type', response.headers['content-type']);
    res.setHeader('Content-Disposition', response.headers['content-disposition']);
    
    // Stream the response
    response.data.pipe(res);
  } catch (error) {
    console.error('Error exporting dataset:', error);
    res.status(500).json({ error: 'Failed to export dataset' });
  }
});

// ML training endpoints
router.post('/ml/train', async (req, res) => {
  try {
    const response = await axios.post(`${SCRAPPER_URL}/api/ml/train`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error starting training:', error);
    res.status(500).json({ error: 'Failed to start training' });
  }
});

router.get('/ml/models', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/ml/models`);
    res.json(response.data);
  } catch (error) {
    console.error('Error listing models:', error);
    res.status(500).json({ error: 'Failed to list models' });
  }
});

router.get('/ml/models/:id', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/ml/models/${req.params.id}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching model:', error);
    res.status(500).json({ error: 'Failed to fetch model' });
  }
});

router.post('/ml/models/:id/evaluate', async (req, res) => {
  try {
    const response = await axios.post(
      `${SCRAPPER_URL}/api/ml/models/${req.params.id}/evaluate`,
      req.body
    );
    res.json(response.data);
  } catch (error) {
    console.error('Error evaluating model:', error);
    res.status(500).json({ error: 'Failed to evaluate model' });
  }
});

// Historical data endpoints
router.get('/data/blocks', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/data/blocks`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching blocks:', error);
    res.status(500).json({ error: 'Failed to fetch blocks' });
  }
});

router.get('/data/transactions', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/data/transactions`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching transactions:', error);
    res.status(500).json({ error: 'Failed to fetch transactions' });
  }
});

router.get('/data/accounts', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/data/accounts`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching accounts:', error);
    res.status(500).json({ error: 'Failed to fetch accounts' });
  }
});

router.get('/data/programs', async (req, res) => {
  try {
    const response = await axios.get(`${SCRAPPER_URL}/api/data/programs`, {
      params: req.query
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching programs:', error);
    res.status(500).json({ error: 'Failed to fetch programs' });
  }
});

// SSE endpoint for real-time progress updates
router.get('/scrapper/progress/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*'
  });
  
  // Function to send SSE
  const sendSSE = (data: any) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };
  
  // Poll for updates every second
  const interval = setInterval(async () => {
    try {
      const response = await axios.get(`${SCRAPPER_URL}/api/scrapper/progress`);
      sendSSE(response.data);
    } catch (error) {
      console.error('Error fetching progress for SSE:', error);
    }
  }, 1000);
  
  // Clean up on client disconnect
  req.on('close', () => {
    clearInterval(interval);
    res.end();
  });
});

export default router;