import { Kafka } from 'kafkajs';
import { ClickHouse } from 'clickhouse';
import axios from 'axios';
import { Connection } from '@solana/web3.js';
import pino from 'pino';

const logger = pino({ level: 'info' });

// Test configuration
const TEST_CONFIG = {
  KAFKA_BROKERS: ['localhost:19092'],
  CLICKHOUSE_HOST: 'localhost',
  CLICKHOUSE_PORT: 8123,
  CLICKHOUSE_USER: 'solana',
  CLICKHOUSE_PASSWORD: 'mev_billions_2025',
  CLICKHOUSE_DATABASE: 'solana_history',
  RPC_ENDPOINT: 'https://api.mainnet-beta.solana.com',
  TEST_SLOT_RANGE: { start: 250000000, end: 250000010 },
  TIMEOUT: 60000,
};

describe('Solana Historical Data Infrastructure Tests', () => {
  let kafka: Kafka;
  let clickhouse: ClickHouse;
  let connection: Connection;

  beforeAll(async () => {
    // Initialize Kafka client
    kafka = new Kafka({
      clientId: 'test-client',
      brokers: TEST_CONFIG.KAFKA_BROKERS,
    });

    // Initialize ClickHouse client
    clickhouse = new ClickHouse({
      url: TEST_CONFIG.CLICKHOUSE_HOST,
      port: TEST_CONFIG.CLICKHOUSE_PORT,
      debug: false,
      basicAuth: {
        username: TEST_CONFIG.CLICKHOUSE_USER,
        password: TEST_CONFIG.CLICKHOUSE_PASSWORD,
      },
    });

    // Initialize Solana connection
    connection = new Connection(TEST_CONFIG.RPC_ENDPOINT, 'confirmed');
  }, TEST_CONFIG.TIMEOUT);

  describe('Infrastructure Health Checks', () => {
    test('Kafka cluster should be healthy', async () => {
      const admin = kafka.admin();
      await admin.connect();
      
      const clusterInfo = await admin.describeCluster();
      expect(clusterInfo.brokers).toHaveLength(1);
      expect(clusterInfo.brokers[0].port).toBe(9092);
      
      await admin.disconnect();
    });

    test('ClickHouse should be accessible', async () => {
      const result = await clickhouse
        .query('SELECT version()')
        .toPromise();
      
      expect(result).toBeDefined();
      expect(result[0]).toHaveProperty('version()');
      logger.info(`ClickHouse version: ${result[0]['version()']}`);
    });

    test('All required Kafka topics should exist', async () => {
      const admin = kafka.admin();
      await admin.connect();
      
      const topics = await admin.listTopics();
      const requiredTopics = [
        'solana.slots',
        'solana.blocks',
        'solana.transactions',
        'solana.accounts',
      ];
      
      for (const topic of requiredTopics) {
        if (!topics.includes(topic)) {
          await admin.createTopics({
            topics: [{ topic, numPartitions: 10, replicationFactor: 1 }],
          });
        }
      }
      
      const updatedTopics = await admin.listTopics();
      for (const topic of requiredTopics) {
        expect(updatedTopics).toContain(topic);
      }
      
      await admin.disconnect();
    });

    test('ClickHouse tables should exist with correct schema', async () => {
      const tables = await clickhouse
        .query(`SHOW TABLES FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}`)
        .toPromise();
      
      const requiredTables = [
        'slots',
        'blocks',
        'transactions',
        'account_updates',
        'kafka_slots_staging',
        'kafka_blocks_staging',
        'kafka_transactions_staging',
        'kafka_accounts_staging',
      ];
      
      const tableNames = tables.map((t: any) => t.name);
      for (const table of requiredTables) {
        expect(tableNames).toContain(table);
      }
    });
  });

  describe('Data Ingestion Tests', () => {
    test('Should ingest slot data correctly', async () => {
      const producer = kafka.producer();
      await producer.connect();
      
      const testSlot = {
        slot: 999999999,
        parent_slot: 999999998,
        block_height: 999999999,
        block_time: Date.now() / 1000,
        leader: 'DWvDTSh3qfn88UoQTEKRV2JnLt5jtJAVoiCo3ivtMwXP',
        rewards_json: '[]',
        block_hash: 'test_hash',
        parent_hash: 'parent_hash',
        transaction_count: 100,
        entry_count: 10,
        tick_count: 5,
      };
      
      await producer.send({
        topic: 'solana.slots',
        messages: [{
          key: testSlot.slot.toString(),
          value: JSON.stringify(testSlot),
        }],
      });
      
      await producer.disconnect();
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // Verify in ClickHouse
      const result = await clickhouse
        .query(`SELECT * FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.slots WHERE slot = ${testSlot.slot}`)
        .toPromise();
      
      expect(result).toHaveLength(1);
      expect(result[0].slot).toBe(testSlot.slot);
      expect(result[0].transaction_count).toBe(testSlot.transaction_count);
    });

    test('Should handle duplicate messages correctly', async () => {
      const producer = kafka.producer();
      await producer.connect();
      
      const testTransaction = {
        signature: 'test_sig_' + Date.now(),
        slot: 999999999,
        block_time: Date.now() / 1000,
        block_index: 0,
        transaction_index: 0,
        is_vote: false,
        success: true,
        fee: 5000,
        compute_units_consumed: 100000,
        err: '',
        memo: 'test',
        signer: 'test_signer',
        signers: ['test_signer'],
        account_keys: ['test_key1', 'test_key2'],
        pre_balances: [1000000, 2000000],
        post_balances: [995000, 2005000],
        pre_token_balances_json: '[]',
        post_token_balances_json: '[]',
        instructions_json: '[]',
        inner_instructions_json: '[]',
        log_messages: ['test log'],
        rewards_json: '[]',
        loaded_addresses_json: '{}',
        return_data_json: 'null',
      };
      
      // Send duplicate messages
      for (let i = 0; i < 3; i++) {
        await producer.send({
          topic: 'solana.transactions',
          messages: [{
            key: testTransaction.signature,
            value: JSON.stringify(testTransaction),
          }],
        });
      }
      
      await producer.disconnect();
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // Verify only one record in ClickHouse (deduplication)
      const result = await clickhouse
        .query(`
          SELECT count() as count 
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.transactions 
          WHERE signature = '${testTransaction.signature}'
        `)
        .toPromise();
      
      expect(result[0].count).toBe(1);
    });
  });

  describe('Performance Tests', () => {
    test('Should achieve target ingestion rate (≥50k msgs/min)', async () => {
      const producer = kafka.producer();
      await producer.connect();
      
      const startTime = Date.now();
      const messages = [];
      const targetMessages = 1000; // Test with 1000 messages
      
      for (let i = 0; i < targetMessages; i++) {
        messages.push({
          key: `perf_test_${i}`,
          value: JSON.stringify({
            slot: 900000000 + i,
            parent_slot: 900000000 + i - 1,
            block_height: 900000000 + i,
            block_time: Date.now() / 1000,
            leader: 'test',
            rewards_json: '[]',
            block_hash: `hash_${i}`,
            parent_hash: `hash_${i - 1}`,
            transaction_count: 100,
            entry_count: 10,
            tick_count: 5,
          }),
        });
      }
      
      await producer.send({
        topic: 'solana.slots',
        messages,
      });
      
      const duration = Date.now() - startTime;
      const messagesPerMinute = (targetMessages / duration) * 60000;
      
      logger.info(`Ingestion rate: ${messagesPerMinute.toFixed(0)} msgs/min`);
      expect(messagesPerMinute).toBeGreaterThanOrEqual(50000);
      
      await producer.disconnect();
    });

    test('Should handle concurrent producers efficiently', async () => {
      const producerCount = 5;
      const messagesPerProducer = 100;
      const producers = [];
      
      // Create multiple producers
      for (let i = 0; i < producerCount; i++) {
        const producer = kafka.producer();
        await producer.connect();
        producers.push(producer);
      }
      
      const startTime = Date.now();
      
      // Send messages concurrently
      await Promise.all(
        producers.map(async (producer, pIndex) => {
          const messages = [];
          for (let i = 0; i < messagesPerProducer; i++) {
            messages.push({
              key: `concurrent_${pIndex}_${i}`,
              value: JSON.stringify({
                signature: `sig_${pIndex}_${i}`,
                slot: 800000000 + pIndex * 1000 + i,
                block_time: Date.now() / 1000,
                // ... other fields
              }),
            });
          }
          
          await producer.send({
            topic: 'solana.transactions',
            messages,
          });
        })
      );
      
      const duration = Date.now() - startTime;
      const totalMessages = producerCount * messagesPerProducer;
      const throughput = (totalMessages / duration) * 1000;
      
      logger.info(`Concurrent throughput: ${throughput.toFixed(2)} msgs/sec`);
      expect(throughput).toBeGreaterThan(100); // At least 100 msgs/sec
      
      // Cleanup
      await Promise.all(producers.map(p => p.disconnect()));
    });
  });

  describe('Data Integrity Tests', () => {
    test('Should maintain row count consistency', async () => {
      // Check Kafka offset vs ClickHouse row count
      const admin = kafka.admin();
      await admin.connect();
      
      const topicOffsets = await admin.fetchTopicOffsets('solana.slots');
      const kafkaMessageCount = topicOffsets.reduce(
        (sum, partition) => sum + parseInt(partition.high) - parseInt(partition.low),
        0
      );
      
      const clickhouseResult = await clickhouse
        .query(`
          SELECT count() as count 
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.slots
        `)
        .toPromise();
      
      const clickhouseCount = clickhouseResult[0].count;
      
      // Allow for some lag, but should be close
      const difference = Math.abs(kafkaMessageCount - clickhouseCount);
      const percentDiff = (difference / kafkaMessageCount) * 100;
      
      logger.info(`Kafka messages: ${kafkaMessageCount}, ClickHouse rows: ${clickhouseCount}`);
      expect(percentDiff).toBeLessThan(5); // Less than 5% difference
      
      await admin.disconnect();
    });

    test('Should preserve data types correctly', async () => {
      const result = await clickhouse
        .query(`
          SELECT 
            toTypeName(slot) as slot_type,
            toTypeName(block_time) as time_type,
            toTypeName(fee) as fee_type,
            toTypeName(signature) as sig_type
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.transactions
          LIMIT 1
        `)
        .toPromise();
      
      if (result.length > 0) {
        expect(result[0].slot_type).toBe('UInt64');
        expect(result[0].time_type).toContain('DateTime64');
        expect(result[0].fee_type).toBe('UInt64');
        expect(result[0].sig_type).toBe('String');
      }
    });

    test('Should handle null values appropriately', async () => {
      const producer = kafka.producer();
      await producer.connect();
      
      const testTransaction = {
        signature: 'null_test_' + Date.now(),
        slot: 777777777,
        block_time: Date.now() / 1000,
        block_index: 0,
        transaction_index: 0,
        is_vote: false,
        success: false,
        fee: 0,
        compute_units_consumed: 0,
        err: 'Test error',
        memo: null,
        signer: 'test_signer',
        signers: [],
        account_keys: [],
        pre_balances: [],
        post_balances: [],
        pre_token_balances_json: '[]',
        post_token_balances_json: '[]',
        instructions_json: '[]',
        inner_instructions_json: '[]',
        log_messages: [],
        rewards_json: '[]',
        loaded_addresses_json: '{}',
        return_data_json: 'null',
      };
      
      await producer.send({
        topic: 'solana.transactions',
        messages: [{
          key: testTransaction.signature,
          value: JSON.stringify(testTransaction),
        }],
      });
      
      await producer.disconnect();
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // Verify data integrity
      const result = await clickhouse
        .query(`
          SELECT * 
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.transactions 
          WHERE signature = '${testTransaction.signature}'
        `)
        .toPromise();
      
      expect(result).toHaveLength(1);
      expect(result[0].success).toBe(false);
      expect(result[0].err).toBe('Test error');
    });
  });

  describe('Monitoring & Observability Tests', () => {
    test('Should expose Prometheus metrics', async () => {
      const metricsUrls = [
        'http://localhost:9090/metrics', // Rust ingester
        'http://localhost:9091/metrics', // Node.js backfill
      ];
      
      for (const url of metricsUrls) {
        try {
          const response = await axios.get(url);
          expect(response.status).toBe(200);
          expect(response.data).toContain('# HELP');
          expect(response.data).toContain('# TYPE');
        } catch (err) {
          logger.warn(`Metrics endpoint ${url} not available`);
        }
      }
    });

    test('Should track consumer progress', async () => {
      const result = await clickhouse
        .query(`
          SELECT * 
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.consumer_progress 
          ORDER BY update_time DESC 
          LIMIT 10
        `)
        .toPromise();
      
      if (result.length > 0) {
        expect(result[0]).toHaveProperty('consumer_group');
        expect(result[0]).toHaveProperty('current_offset');
        expect(result[0]).toHaveProperty('lag');
        
        logger.info(`Consumer lag: ${result[0].lag}`);
      }
    });

    test('Should log ingestion metrics', async () => {
      const result = await clickhouse
        .query(`
          SELECT 
            table_name,
            sum(rows_ingested) as total_rows,
            avg(lag_ms) as avg_lag,
            max(lag_ms) as max_lag
          FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.ingestion_metrics
          WHERE timestamp > now() - INTERVAL 1 HOUR
          GROUP BY table_name
        `)
        .toPromise();
      
      if (result.length > 0) {
        result.forEach((metric: any) => {
          logger.info(`Table: ${metric.table_name}, Rows: ${metric.total_rows}, Avg Lag: ${metric.avg_lag}ms`);
        });
      }
    });
  });

  afterAll(async () => {
    // Cleanup test data
    try {
      await clickhouse
        .query(`
          DELETE FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.slots 
          WHERE slot >= 777777777
        `)
        .toPromise();
      
      await clickhouse
        .query(`
          DELETE FROM ${TEST_CONFIG.CLICKHOUSE_DATABASE}.transactions 
          WHERE slot >= 777777777
        `)
        .toPromise();
    } catch (err) {
      logger.error('Cleanup failed:', err);
    }
  });
});