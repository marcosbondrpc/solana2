# 🌐 Network Access Configuration

## ✅ All Services Configured for Remote Access

All services are now configured to accept connections from any network interface (0.0.0.0), allowing access from other machines on your network or internet.

## 📡 Your Server Information

- **Server IP**: `45.157.234.184`
- **Network Binding**: `0.0.0.0` (all interfaces)
- **Access Mode**: Remote access enabled

## 🔗 Service URLs for Remote Access

Access these services from any machine using your server's IP address:

### Infrastructure Services

| Service | Local URL | Remote URL | Credentials |
|---------|-----------|------------|-------------|
| **Grafana Dashboard** | http://localhost:3001 | http://45.157.234.184:3001 | admin/admin |
| **ClickHouse** | http://localhost:8123 | http://45.157.234.184:8123 | default/arbitrage123 |
| **Prometheus** | http://localhost:9090 | http://45.157.234.184:9090 | - |
| **Redis** | redis://localhost:6390 | redis://45.157.234.184:6379 | - |
| **Kafka** | localhost:9092 | 45.157.234.184:9092 | - |
| **Zookeeper** | localhost:2181 | 45.157.234.184:2181 | - |

### Application Services

| Service | Local URL | Remote URL | Description |
|---------|-----------|------------|-------------|
| **Frontend Dashboard** | http://localhost:3000 | http://45.157.234.184:3000 | React Dashboard |
| **Backend API** | http://localhost:8000 | http://45.157.234.184:8000 | Control Plane API |
| **API Documentation** | http://localhost:8000/docs | http://45.157.234.184:8000/docs | Swagger UI |

## 🚀 Starting Services for Network Access

### 1. Infrastructure (Already Running)
```bash
# All Docker services are already configured and running
# Check status:
sudo docker ps

# If needed, restart:
sudo docker compose -f arbitrage-data-capture/docker-compose.yml restart
```

### 2. Frontend Dashboard
```bash
cd frontend2
npm install
npm run dev
# Will be accessible at http://45.157.234.184:3000
```

### 3. Backend API
```bash
cd backend/services/control-plane
pip3 install fastapi uvicorn
python3 main.py
# Will be accessible at http://45.157.234.184:8000
# API docs at http://45.157.234.184:8000/docs
```

## 🔒 Security Configuration

### Current Settings (Development Mode)

All services are configured for **open access** suitable for development:

- **CORS**: Allow all origins (*)
- **Authentication**: Disabled for testing
- **Redis**: Protected mode disabled
- **Network Binding**: 0.0.0.0 (all interfaces)

### Firewall Configuration

If you have a firewall, ensure these ports are open:

```bash
# For UFW (Ubuntu Firewall)
sudo ufw allow 3000/tcp  # Frontend
sudo ufw allow 3001/tcp  # Grafana
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 8123/tcp  # ClickHouse HTTP
sudo ufw allow 9000/tcp  # ClickHouse Native
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 6379/tcp  # Redis
sudo ufw allow 9092/tcp  # Kafka
sudo ufw allow 2181/tcp  # Zookeeper

# Or allow all at once
for port in 3000 3001 8000 8123 9000 9090 6379 9092 2181; do
  sudo ufw allow $port/tcp
done
```

### For Cloud Providers (AWS/GCP/Azure)

Add inbound rules to your security group/firewall for these ports:

- **3000** - Frontend Dashboard
- **3001** - Grafana
- **8000** - Backend API
- **8123** - ClickHouse HTTP
- **9000** - ClickHouse Native
- **9090** - Prometheus
- **6379** - Redis (⚠️ Be careful with Redis in production)
- **9092** - Kafka
- **2181** - Zookeeper

## 🧪 Testing Remote Access

### From Another Machine

1. **Test Grafana**:
   ```bash
   curl http://45.157.234.184:3001
   # Or open in browser: http://45.157.234.184:3001
   ```

2. **Test ClickHouse**:
   ```bash
   curl http://45.157.234.184:8123/
   # Should return: Ok.
   ```

3. **Test Backend API** (after starting):
   ```bash
   curl http://45.157.234.184:8000/health
   # Should return JSON health status
   ```

4. **Test Redis**:
   ```bash
   redis-cli -h 45.157.234.184 ping
   # Should return: PONG
   ```

## 📱 Mobile/Tablet Access

You can access the dashboards from mobile devices on the same network:

1. Open your mobile browser
2. Navigate to: `http://45.157.234.184:3001` (Grafana)
3. Or: `http://45.157.234.184:3000` (Frontend Dashboard - after starting)

## ⚠️ Production Security Recommendations

**WARNING**: Current configuration is for DEVELOPMENT only!

For production, you should:

1. **Enable Authentication**:
   - Set strong passwords for all services
   - Enable Redis password protection
   - Configure JWT authentication for APIs

2. **Restrict Network Access**:
   - Use specific IP allowlists instead of 0.0.0.0
   - Configure firewall rules
   - Use VPN for remote access

3. **Use HTTPS/TLS**:
   - Configure SSL certificates
   - Use reverse proxy (nginx/traefik)
   - Enable TLS for Redis and databases

4. **Monitor Access**:
   - Enable audit logging
   - Monitor failed authentication attempts
   - Set up intrusion detection

## 🔧 Troubleshooting

### Cannot Access from Remote Machine

1. **Check if services are running**:
   ```bash
   sudo docker ps
   netstat -tulpn | grep LISTEN
   ```

2. **Check firewall**:
   ```bash
   sudo ufw status
   sudo iptables -L
   ```

3. **Test connectivity**:
   ```bash
   # From remote machine
   ping 45.157.234.184
   telnet 45.157.234.184 3001
   ```

4. **Check Docker binding**:
   ```bash
   sudo docker port grafana
   # Should show: 3000/tcp -> 0.0.0.0:3001
   ```

### Service Not Accessible

1. **Verify port binding**:
   ```bash
   sudo netstat -tulpn | grep <port>
   ```

2. **Check Docker logs**:
   ```bash
   sudo docker logs <container-name>
   ```

3. **Restart service**:
   ```bash
   sudo docker restart <container-name>
   ```

## 📋 Quick Reference

### Access URLs from Any Device

```
Grafana:     http://45.157.234.184:3001
ClickHouse:  http://45.157.234.184:8123
Prometheus:  http://45.157.234.184:9090
Frontend:    http://45.157.234.184:3000 (after npm run dev)
Backend API: http://45.157.234.184:8000 (after python3 main.py)
API Docs:    http://45.157.234.184:8000/docs
```

### Connection Strings for Applications

```javascript
// Frontend .env
VITE_API_URL=http://45.157.234.184:8000
VITE_WS_URL=ws://45.157.234.184:8001
VITE_CLICKHOUSE_URL=http://45.157.234.184:8123

// Backend configuration
REDIS_URL=redis://45.157.234.184:6379
CLICKHOUSE_URL=http://45.157.234.184:8123
KAFKA_BROKERS=45.157.234.184:9092
```

## ✅ Status

All services are configured and ready for network access! You can now access your Solana MEV Infrastructure from any device on your network or from the internet (if ports are exposed).

---

**Network Configuration**: ✅ COMPLETE
**Remote Access**: ✅ ENABLED
**Server IP**: `45.157.234.184`