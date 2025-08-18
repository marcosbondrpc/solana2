/**
 * @fileoverview Ultra-fast message validation for MEV systems
 * 
 * Cryptographic validation and message integrity checks with
 * microsecond-level performance for high-frequency trading.
 */

import { globalMetrics } from '../index';

/**
 * Message validation result
 */
export interface ValidationResult {
  readonly valid: boolean;
  readonly errorCode?: string;
  readonly errorMessage?: string;
  readonly validationTimeUs?: number;
}

/**
 * Fast Ed25519 signature validator
 */
export class Ed25519Validator {
  private readonly publicKeys = new Map<string, CryptoKey>();

  /**
   * Import and cache public key
   */
  async importPublicKey(keyId: string, publicKeyBytes: Uint8Array): Promise<void> {
    try {
      const cryptoKey = await crypto.subtle.importKey(
        'raw',
        publicKeyBytes,
        {
          name: 'Ed25519',
          namedCurve: 'Ed25519'
        },
        false,
        ['verify']
      );
      this.publicKeys.set(keyId, cryptoKey);
    } catch (error) {
      throw new Error(`Failed to import public key ${keyId}: ${error}`);
    }
  }

  /**
   * Verify message signature with cached key
   */
  async verifySignature(
    keyId: string,
    message: Uint8Array,
    signature: Uint8Array
  ): Promise<ValidationResult> {
    const startTime = performance.now();

    try {
      const publicKey = this.publicKeys.get(keyId);
      if (!publicKey) {
        return {
          valid: false,
          errorCode: 'KEY_NOT_FOUND',
          errorMessage: `Public key not found for ID: ${keyId}`,
          validationTimeUs: (performance.now() - startTime) * 1000
        };
      }

      const isValid = await crypto.subtle.verify(
        'Ed25519',
        publicKey,
        signature,
        message
      );

      const validationTimeUs = (performance.now() - startTime) * 1000;
      globalMetrics.recordParseLatency(validationTimeUs);

      return {
        valid: isValid,
        validationTimeUs
      };
    } catch (error) {
      return {
        valid: false,
        errorCode: 'SIGNATURE_VERIFICATION_FAILED',
        errorMessage: `Signature verification failed: ${error}`,
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }
  }

  /**
   * Batch verify multiple signatures
   */
  async verifyBatch(
    verifications: Array<{
      keyId: string;
      message: Uint8Array;
      signature: Uint8Array;
    }>
  ): Promise<ValidationResult[]> {
    const promises = verifications.map(v => 
      this.verifySignature(v.keyId, v.message, v.signature)
    );
    return Promise.all(promises);
  }

  /**
   * Clear cached keys
   */
  clearCache(): void {
    this.publicKeys.clear();
  }
}

/**
 * Fast message integrity validator
 */
export class MessageValidator {
  private readonly maxMessageAge = 60_000_000_000; // 60 seconds in nanoseconds
  private readonly nonceCache = new Set<string>();
  private readonly maxNonceCache = 100000;

  /**
   * Validate message envelope
   */
  validateEnvelope(envelope: any): ValidationResult {
    const startTime = performance.now();

    // Check required fields
    if (!envelope.timestamp_ns) {
      return {
        valid: false,
        errorCode: 'MISSING_TIMESTAMP',
        errorMessage: 'Missing timestamp_ns field',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (!envelope.stream_id || typeof envelope.stream_id !== 'string') {
      return {
        valid: false,
        errorCode: 'INVALID_STREAM_ID',
        errorMessage: 'Invalid or missing stream_id',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (typeof envelope.sequence !== 'number' || envelope.sequence < 0) {
      return {
        valid: false,
        errorCode: 'INVALID_SEQUENCE',
        errorMessage: 'Invalid sequence number',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    // Check message age
    const now = BigInt(Date.now() * 1_000_000); // Convert to nanoseconds
    const messageTime = BigInt(envelope.timestamp_ns);
    const age = now - messageTime;

    if (age > this.maxMessageAge) {
      return {
        valid: false,
        errorCode: 'MESSAGE_TOO_OLD',
        errorMessage: `Message is ${age / 1_000_000n}ms old`,
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (age < 0) {
      return {
        valid: false,
        errorCode: 'MESSAGE_FROM_FUTURE',
        errorMessage: 'Message timestamp is in the future',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    return {
      valid: true,
      validationTimeUs: (performance.now() - startTime) * 1000
    };
  }

  /**
   * Validate control command with anti-replay protection
   */
  validateCommand(command: any): ValidationResult {
    const startTime = performance.now();

    // Validate envelope first
    const envelopeResult = this.validateEnvelope(command);
    if (!envelopeResult.valid) {
      return envelopeResult;
    }

    // Check command-specific fields
    if (!command.id || typeof command.id !== 'string') {
      return {
        valid: false,
        errorCode: 'INVALID_COMMAND_ID',
        errorMessage: 'Invalid or missing command ID',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (!command.module || typeof command.module !== 'string') {
      return {
        valid: false,
        errorCode: 'INVALID_MODULE',
        errorMessage: 'Invalid or missing module',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (!command.action || typeof command.action !== 'string') {
      return {
        valid: false,
        errorCode: 'INVALID_ACTION',
        errorMessage: 'Invalid or missing action',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    // Anti-replay protection
    if (typeof command.nonce === 'number') {
      const nonceKey = `${command.pubkey_id}:${command.nonce}`;
      if (this.nonceCache.has(nonceKey)) {
        return {
          valid: false,
          errorCode: 'REPLAY_ATTACK',
          errorMessage: 'Nonce has been used before',
          validationTimeUs: (performance.now() - startTime) * 1000
        };
      }
      
      // Add to nonce cache
      this.nonceCache.add(nonceKey);
      
      // Prevent unbounded growth
      if (this.nonceCache.size > this.maxNonceCache) {
        // Remove oldest entries (simple FIFO)
        const iterator = this.nonceCache.values();
        for (let i = 0; i < 1000; i++) {
          const value = iterator.next().value;
          if (value) this.nonceCache.delete(value);
        }
      }
    }

    return {
      valid: true,
      validationTimeUs: (performance.now() - startTime) * 1000
    };
  }

  /**
   * Validate MEV opportunity message
   */
  validateMevOpportunity(opportunity: any): ValidationResult {
    const startTime = performance.now();

    if (!opportunity.tx_hash || typeof opportunity.tx_hash !== 'string') {
      return {
        valid: false,
        errorCode: 'INVALID_TX_HASH',
        errorMessage: 'Invalid transaction hash',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (typeof opportunity.slot !== 'number' || opportunity.slot < 0) {
      return {
        valid: false,
        errorCode: 'INVALID_SLOT',
        errorMessage: 'Invalid slot number',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (typeof opportunity.profit_lamports !== 'number') {
      return {
        valid: false,
        errorCode: 'INVALID_PROFIT',
        errorMessage: 'Invalid profit amount',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    if (typeof opportunity.probability !== 'number' || 
        opportunity.probability < 0 || opportunity.probability > 1) {
      return {
        valid: false,
        errorCode: 'INVALID_PROBABILITY',
        errorMessage: 'Invalid probability (must be 0-1)',
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    return {
      valid: true,
      validationTimeUs: (performance.now() - startTime) * 1000
    };
  }

  /**
   * Batch validate multiple messages
   */
  validateBatch(messages: any[], validationType: 'envelope' | 'command' | 'opportunity'): ValidationResult[] {
    return messages.map(msg => {
      switch (validationType) {
        case 'envelope':
          return this.validateEnvelope(msg);
        case 'command':
          return this.validateCommand(msg);
        case 'opportunity':
          return this.validateMevOpportunity(msg);
        default:
          return {
            valid: false,
            errorCode: 'UNKNOWN_VALIDATION_TYPE',
            errorMessage: `Unknown validation type: ${validationType}`
          };
      }
    });
  }

  /**
   * Clear nonce cache for testing
   */
  clearNonceCache(): void {
    this.nonceCache.clear();
  }
}

/**
 * Schema validator for structured data
 */
export class SchemaValidator {
  private readonly schemas = new Map<string, any>();

  /**
   * Register a JSON schema
   */
  registerSchema(name: string, schema: any): void {
    this.schemas.set(name, schema);
  }

  /**
   * Validate data against registered schema
   */
  validate(schemaName: string, data: any): ValidationResult {
    const startTime = performance.now();
    const schema = this.schemas.get(schemaName);

    if (!schema) {
      return {
        valid: false,
        errorCode: 'SCHEMA_NOT_FOUND',
        errorMessage: `Schema not found: ${schemaName}`,
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }

    // Simple validation - in production, use a proper JSON schema validator
    try {
      const isValid = this.validateObject(data, schema);
      return {
        valid: isValid,
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    } catch (error) {
      return {
        valid: false,
        errorCode: 'VALIDATION_ERROR',
        errorMessage: `Validation failed: ${error}`,
        validationTimeUs: (performance.now() - startTime) * 1000
      };
    }
  }

  private validateObject(data: any, schema: any): boolean {
    // Simplified validation - implement full JSON schema validation in production
    if (schema.type === 'object' && schema.required) {
      for (const field of schema.required) {
        if (!(field in data)) {
          throw new Error(`Missing required field: ${field}`);
        }
      }
    }
    return true;
  }
}

// Global validator instances
export const globalSignatureValidator = new Ed25519Validator();
export const globalMessageValidator = new MessageValidator();
export const globalSchemaValidator = new SchemaValidator();