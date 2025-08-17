# Security Documentation

## Overview

This document outlines the security architecture and practices for the MEV monitoring infrastructure. This is a **DEFENSIVE-ONLY** system with no execution or trading capabilities.

## Authentication & Authorization

### JWT Authentication
- Algorithm: RS256 with 2048-bit keys
- Token lifetime: 15 minutes (access), 7 days (refresh)
- Automatic refresh before expiration
- Tokens include user ID, email, roles, and permissions

### RBAC (Role-Based Access Control)

| Role | Permissions | Description |
|------|------------|-------------|
| **viewer** | Read-only access to dashboards | Basic monitoring access |
| **analyst** | Export data, run queries | Data analysis capabilities |
| **operator** | Control kill-switches, view audit logs | System operations |
| **ml_engineer** | Train models, deploy (shadow/canary) | ML pipeline management |
| **admin** | Full system access | Complete administrative control |

### Multi-Factor Authentication (MFA)
- TOTP-based 2FA for sensitive operations
- Required for: admin role, kill-switch activation, model deployment
- 30-second time window with clock skew tolerance

## Cryptographic Controls

### Ed25519 Signatures
Critical operations require Ed25519 signatures:
- Kill-switch activation
- Model deployment
- System configuration changes
- Audit log entries

### 2-of-3 Multisig
System-wide changes require 2-of-3 multisig:
- Emergency stop
- Configuration updates
- Role permission changes

### Hash Chain Audit Trail
- Each audit event includes hash of previous event
- Immutable chain for forensic analysis
- Daily Merkle root anchoring to blockchain

## API Security

### Input Validation
- Pydantic models for all inputs
- SQL injection prevention via parameterized queries
- XSS protection via content security policy
- Request size limits (10MB default)

### Rate Limiting
Per-role rate limits:
- viewer: 60 requests/minute
- analyst: 120 requests/minute
- operator: 300 requests/minute
- ml_engineer: 300 requests/minute
- admin: 600 requests/minute

### CORS Configuration
```python
origins = [
    "http://localhost:3001",
    "http://localhost:5173",
    "https://mev-dashboard.example.com"
]
```

## Database Security

### ClickHouse
- Read-only user for queries
- Parameterized queries only
- Table/column whitelisting
- Query timeout: 30 seconds
- Result size limit: 100MB

### Access Control
- Separate users for read/write operations
- Network isolation via Docker networks
- TLS encryption for connections
- Regular password rotation

## WebSocket Security

### Connection Authentication
- JWT token required in connection params
- Token validation before upgrade
- Automatic disconnection on token expiry

### Message Security
- Binary protobuf encoding
- Message size limits (1MB)
- Topic-based permissions
- Rate limiting per connection

## Infrastructure Security

### Docker Security
- Non-root containers
- Read-only root filesystems where possible
- Security scanning of images
- Network segmentation
- Resource limits

### Network Security
- TLS 1.3 for all external connections
- Internal communication via Docker networks
- Firewall rules for port access
- DDoS protection via rate limiting

## Monitoring & Alerting

### Security Metrics
- Failed authentication attempts
- Unusual query patterns
- Rate limit violations
- Signature verification failures
- Audit trail gaps

### Alert Triggers
- 5+ failed auth attempts in 1 minute
- Query accessing restricted tables
- Kill-switch activation
- Multisig threshold not met
- Hash chain discontinuity

## Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|------------|--------------|---------|
| **Critical** | System compromise | < 15 minutes | Unauthorized admin access |
| **High** | Security breach attempt | < 1 hour | Multiple signature failures |
| **Medium** | Policy violation | < 4 hours | Excessive query usage |
| **Low** | Minor issue | < 24 hours | Failed health check |

### Response Procedures

1. **Detection**
   - Automated alerts via Prometheus/Grafana
   - Audit log analysis
   - User reports

2. **Containment**
   - Activate kill-switches if needed
   - Revoke compromised tokens
   - Block suspicious IPs

3. **Investigation**
   - Review audit trail
   - Analyze system logs
   - Check hash chain integrity

4. **Recovery**
   - Restore from backups if needed
   - Reset affected credentials
   - Update security rules

5. **Post-Mortem**
   - Document incident
   - Update procedures
   - Implement preventive measures

## Best Practices

### Development
- Security review for all PRs
- Dependency vulnerability scanning
- Static code analysis
- Penetration testing quarterly

### Operations
- Principle of least privilege
- Regular security audits
- Automated vulnerability scanning
- Security training for team

### Data Protection
- Encryption at rest and in transit
- PII minimization
- Data retention policies
- GDPR compliance where applicable

## Compliance

### Standards
- SOC 2 Type II (planned)
- ISO 27001 (planned)
- OWASP Top 10 mitigation

### Audit Requirements
- All actions logged with user, timestamp, and result
- Logs retained for 1 year minimum
- Quarterly security audits
- Annual penetration testing

## Security Contacts

- Security Team: security@example.com
- Incident Response: incident@example.com
- Bug Bounty: bugbounty@example.com

## Security Checklist

### Daily
- [ ] Review failed authentication attempts
- [ ] Check rate limit violations
- [ ] Verify hash chain continuity

### Weekly
- [ ] Review audit logs for anomalies
- [ ] Check for dependency updates
- [ ] Verify backup integrity

### Monthly
- [ ] Rotate service credentials
- [ ] Review user permissions
- [ ] Security metrics review

### Quarterly
- [ ] Penetration testing
- [ ] Security training
- [ ] Policy review and update