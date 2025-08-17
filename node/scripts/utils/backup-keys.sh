#!/bin/bash
# Backup validator keys and configuration

set -e

BACKUP_DIR="/home/kidgordones/0solana/node/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="keys_backup_${TIMESTAMP}"

echo "Creating backup of keys and configuration..."

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup keys
if [ -d "/home/kidgordones/0solana/node/keys" ]; then
    cp -r /home/kidgordones/0solana/node/keys "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo "✓ Keys backed up"
fi

# Backup configs
if [ -d "/home/kidgordones/0solana/node/configs" ]; then
    cp -r /home/kidgordones/0solana/node/configs "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo "✓ Configs backed up"
fi

# Backup identity files
if [ -f "/opt/solana/identity.json" ]; then
    cp /opt/solana/identity.json "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo "✓ Identity keypair backed up"
fi

if [ -f "/home/solana/validator-keypair.json" ]; then
    cp /home/solana/validator-keypair.json "${BACKUP_DIR}/${BACKUP_NAME}/"
    echo "✓ Validator keypair backed up"
fi

# Create tarball
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

echo ""
echo "Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "IMPORTANT: Store this backup in a secure, offline location!"