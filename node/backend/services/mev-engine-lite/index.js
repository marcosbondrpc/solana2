const express = require('express');
const cors = require('cors');
const app = express();
app.use(cors());
app.use(express.json());

// MEV metrics endpoint
app.get('/api/node/metrics', (req, res) => {
  res.json({
    status: 'success',
    data: {
      slot: 280000000 + Math.floor(Date.now() / 250),
      block_height: 250000000 + Math.floor(Date.now() / 500),
      tps: 3000 + Math.sin(Date.now() / 3000) * 500,
      peers: 1000 + Math.floor(Math.random() * 100),
      rpc_latency: 20 + Math.sin(Date.now() / 1000) * 10,
      websocket_latency: 15 + Math.cos(Date.now() / 1000) * 8,
      geyser_latency: 25 + Math.sin(Date.now() / 1500) * 12,
      jito_latency: 18 + Math.cos(Date.now() / 2000) * 9,
      timestamp: Date.now()
    }
  });
});

app.get('/health', (req, res) => res.json({ status: 'healthy' }));

app.listen(8081, () => {
  console.log('MEV Engine Lite running on port 8081');
});
