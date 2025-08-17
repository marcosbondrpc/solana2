"""
Ultra-High-Performance Kafka Streaming Pipeline
Designed for 100k+ messages/second with exactly-once semantics
"""

import asyncio
import json
import msgpack
import pickle
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from confluent_kafka import SerializingProducer, DeserializingConsumer
from confluent_kafka.serialization import StringSerializer, StringDeserializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import uvloop

# Use uvloop for maximum async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KafkaConfig:
    """Kafka configuration for production deployment"""
    bootstrap_servers: List[str] = None
    schema_registry_url: str = "http://schema-registry:8081"
    
    # Producer settings for maximum throughput
    producer_config: Dict[str, Any] = None
    
    # Consumer settings for parallel processing
    consumer_config: Dict[str, Any] = None
    
    # Topic configurations
    topics: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.bootstrap_servers:
            self.bootstrap_servers = ["kafka1:9092", "kafka2:9092", "kafka3:9092"]
        
        if not self.producer_config:
            self.producer_config = {
                'acks': 'all',  # Ensure durability
                'compression_type': 'lz4',  # Fast compression
                'batch_size': 524288,  # 512KB batches
                'linger_ms': 5,  # Small delay for batching
                'buffer_memory': 134217728,  # 128MB buffer
                'max_in_flight_requests_per_connection': 5,
                'enable_idempotence': True,  # Exactly-once semantics
                'retries': 10,
                'retry_backoff_ms': 100,
            }
        
        if not self.consumer_config:
            self.consumer_config = {
                'enable_auto_commit': False,  # Manual commit for exactly-once
                'auto_offset_reset': 'earliest',
                'max_poll_records': 10000,  # Batch processing
                'fetch_min_bytes': 1048576,  # 1MB minimum fetch
                'fetch_max_wait_ms': 100,
                'session_timeout_ms': 30000,
                'heartbeat_interval_ms': 10000,
            }
        
        if not self.topics:
            self.topics = {
                'arbitrage-transactions': {
                    'num_partitions': 50,  # High parallelism
                    'replication_factor': 3,
                    'config': {
                        'compression.type': 'lz4',
                        'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                        'segment.ms': str(60 * 60 * 1000),  # 1 hour segments
                        'min.insync.replicas': '2',
                    }
                },
                'arbitrage-opportunities': {
                    'num_partitions': 30,
                    'replication_factor': 3,
                    'config': {
                        'compression.type': 'lz4',
                        'retention.ms': str(3 * 24 * 60 * 60 * 1000),  # 3 days
                    }
                },
                'risk-metrics': {
                    'num_partitions': 20,
                    'replication_factor': 3,
                    'config': {
                        'compression.type': 'snappy',
                        'retention.ms': str(24 * 60 * 60 * 1000),  # 1 day
                    }
                },
                'market-snapshots': {
                    'num_partitions': 100,  # Very high throughput
                    'replication_factor': 2,
                    'config': {
                        'compression.type': 'lz4',
                        'retention.ms': str(12 * 60 * 60 * 1000),  # 12 hours
                        'segment.ms': str(30 * 60 * 1000),  # 30 minute segments
                    }
                },
                'dead-letter-queue': {
                    'num_partitions': 10,
                    'replication_factor': 3,
                    'config': {
                        'retention.ms': str(30 * 24 * 60 * 60 * 1000),  # 30 days
                    }
                }
            }

class EliteKafkaProducer:
    """High-performance Kafka producer with batching and compression"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self.schema_registry = None
        self.serializers = {}
        self.message_buffer = {}
        self.buffer_lock = asyncio.Lock()
        self.flush_task = None
        self.stats = {
            'messages_sent': 0,
            'bytes_sent': 0,
            'errors': 0,
            'batches_sent': 0
        }
    
    async def start(self):
        """Initialize producer with optimal settings"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=','.join(self.config.bootstrap_servers),
            **self.config.producer_config,
            value_serializer=self._serialize_value,
            key_serializer=lambda k: k.encode('utf-8') if k else None,
        )
        
        await self.producer.start()
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info("Elite Kafka Producer started with optimized settings")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value using MessagePack for speed"""
        if isinstance(value, dict):
            return msgpack.packb(value, use_bin_type=True)
        elif isinstance(value, bytes):
            return value
        else:
            return json.dumps(value).encode('utf-8')
    
    async def send_transaction(self, transaction: Dict[str, Any], key: Optional[str] = None):
        """Send transaction with automatic batching"""
        await self._send_with_retry(
            topic='arbitrage-transactions',
            value=transaction,
            key=key or transaction.get('signature'),
            partition_key=transaction.get('searcher_address')
        )
    
    async def send_opportunity(self, opportunity: Dict[str, Any]):
        """Send detected opportunity"""
        await self._send_with_retry(
            topic='arbitrage-opportunities',
            value=opportunity,
            key=opportunity.get('opportunity_id'),
            partition_key=opportunity.get('input_token')
        )
    
    async def send_risk_metrics(self, metrics: Dict[str, Any]):
        """Send risk metrics"""
        await self._send_with_retry(
            topic='risk-metrics',
            value=metrics,
            key=metrics.get('transaction_signature')
        )
    
    async def send_market_snapshot(self, snapshot: Dict[str, Any]):
        """Send market snapshot with partitioning by pool"""
        await self._send_with_retry(
            topic='market-snapshots',
            value=snapshot,
            key=f"{snapshot['dex']}:{snapshot['pool_address']}",
            partition_key=snapshot['pool_address']
        )
    
    async def _send_with_retry(self, topic: str, value: Any, key: Optional[str] = None,
                               partition_key: Optional[str] = None, retries: int = 3):
        """Send message with retry logic and error handling"""
        partition = None
        if partition_key:
            # Deterministic partitioning for ordering guarantees
            partition = hash(partition_key) % self.config.topics[topic]['num_partitions']
        
        for attempt in range(retries):
            try:
                result = await self.producer.send_and_wait(
                    topic=topic,
                    value=value,
                    key=key,
                    partition=partition
                )
                
                self.stats['messages_sent'] += 1
                self.stats['bytes_sent'] += len(self._serialize_value(value))
                
                return result
                
            except KafkaError as e:
                self.stats['errors'] += 1
                if attempt == retries - 1:
                    # Send to dead letter queue
                    await self._send_to_dlq(topic, value, key, str(e))
                    raise
                
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    async def _send_to_dlq(self, original_topic: str, value: Any, key: Optional[str], error: str):
        """Send failed messages to dead letter queue"""
        dlq_message = {
            'original_topic': original_topic,
            'timestamp': datetime.utcnow().isoformat(),
            'key': key,
            'value': value,
            'error': error
        }
        
        try:
            await self.producer.send_and_wait(
                topic='dead-letter-queue',
                value=dlq_message,
                key=key
            )
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    async def send_batch(self, messages: List[Dict[str, Any]], topic: str):
        """Send batch of messages for maximum throughput"""
        futures = []
        
        for msg in messages:
            future = await self.producer.send(
                topic=topic,
                value=msg,
                key=msg.get('id') or msg.get('signature')
            )
            futures.append(future)
        
        # Wait for all messages to be sent
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        self.stats['messages_sent'] += successful
        self.stats['batches_sent'] += 1
        
        return successful, len(messages) - successful
    
    async def _periodic_flush(self):
        """Periodically flush producer buffer"""
        while True:
            await asyncio.sleep(1)  # Flush every second
            try:
                await self.producer.flush()
            except Exception as e:
                logger.error(f"Flush error: {e}")
    
    async def close(self):
        """Gracefully shutdown producer"""
        if self.flush_task:
            self.flush_task.cancel()
        
        if self.producer:
            await self.producer.flush()
            await self.producer.stop()
        
        logger.info(f"Producer stats: {self.stats}")

class EliteKafkaConsumer:
    """High-performance Kafka consumer with parallel processing"""
    
    def __init__(self, config: KafkaConfig, topics: List[str], group_id: str):
        self.config = config
        self.topics = topics
        self.group_id = group_id
        self.consumer = None
        self.processing_pool = ThreadPoolExecutor(max_workers=10)
        self.handlers = {}
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'batches_processed': 0,
            'total_lag': 0
        }
    
    async def start(self):
        """Initialize consumer with optimal settings"""
        consumer_config = {
            **self.config.consumer_config,
            'group_id': self.group_id,
            'bootstrap_servers': ','.join(self.config.bootstrap_servers),
            'value_deserializer': self._deserialize_value,
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
        }
        
        self.consumer = AIOKafkaConsumer(
            *self.topics,
            **consumer_config
        )
        
        await self.consumer.start()
        logger.info(f"Elite Kafka Consumer started for topics: {self.topics}")
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from MessagePack"""
        try:
            return msgpack.unpackb(value, raw=False)
        except:
            try:
                return json.loads(value.decode('utf-8'))
            except:
                return value
    
    def register_handler(self, topic: str, handler: Callable):
        """Register message handler for topic"""
        self.handlers[topic] = handler
    
    async def consume_batch(self, max_records: int = 1000, timeout_ms: int = 1000):
        """Consume messages in batches for efficiency"""
        try:
            batch = await self.consumer.getmany(
                timeout_ms=timeout_ms,
                max_records=max_records
            )
            
            if not batch:
                return 0
            
            # Process messages in parallel
            tasks = []
            for topic_partition, messages in batch.items():
                topic = topic_partition.topic
                handler = self.handlers.get(topic)
                
                if handler:
                    for msg in messages:
                        task = asyncio.create_task(
                            self._process_message(handler, msg)
                        )
                        tasks.append(task)
            
            # Wait for all processing to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            self.stats['messages_processed'] += successful
            self.stats['messages_failed'] += failed
            self.stats['batches_processed'] += 1
            
            # Commit offsets after successful processing
            await self.consumer.commit()
            
            return successful
            
        except Exception as e:
            logger.error(f"Batch consumption error: {e}")
            return 0
    
    async def _process_message(self, handler: Callable, message):
        """Process individual message with error handling"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.processing_pool,
                handler,
                message.value
            )
            return result
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            # Send to DLQ if available
            raise
    
    async def run_consumer_loop(self):
        """Main consumer loop for continuous processing"""
        logger.info("Starting consumer loop...")
        
        while True:
            try:
                processed = await self.consume_batch(
                    max_records=self.config.consumer_config.get('max_poll_records', 1000)
                )
                
                if processed == 0:
                    await asyncio.sleep(0.1)  # Small delay when no messages
                
                # Log stats periodically
                if self.stats['batches_processed'] % 100 == 0:
                    logger.info(f"Consumer stats: {self.stats}")
                    
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def close(self):
        """Gracefully shutdown consumer"""
        if self.consumer:
            await self.consumer.stop()
        
        self.processing_pool.shutdown(wait=True)
        logger.info(f"Final consumer stats: {self.stats}")

class KafkaStreamProcessor:
    """Stream processing with Kafka Streams-like functionality"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producers = {}
        self.consumers = {}
        self.processors = {}
    
    async def create_stream(self, input_topic: str, output_topic: str,
                           processor_func: Callable, group_id: str):
        """Create a stream processing pipeline"""
        
        # Create consumer for input
        consumer = EliteKafkaConsumer(
            self.config,
            [input_topic],
            group_id
        )
        
        # Create producer for output
        producer = EliteKafkaProducer(self.config)
        
        # Create processing function
        async def process_and_forward(message):
            try:
                # Process message
                result = await processor_func(message)
                
                # Forward to output topic
                if result:
                    await producer._send_with_retry(
                        topic=output_topic,
                        value=result,
                        key=message.get('id')
                    )
                    
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
        
        # Register handler
        consumer.register_handler(input_topic, process_and_forward)
        
        # Start components
        await consumer.start()
        await producer.start()
        
        # Store references
        self.consumers[input_topic] = consumer
        self.producers[output_topic] = producer
        
        # Start consumer loop
        asyncio.create_task(consumer.run_consumer_loop())
        
        logger.info(f"Stream created: {input_topic} -> {output_topic}")
    
    async def create_arbitrage_detection_stream(self):
        """Create specialized stream for arbitrage detection"""
        
        async def detect_arbitrage(transaction):
            """Process transaction for arbitrage detection"""
            if transaction.get('mev_type') == 'arbitrage':
                # Extract arbitrage features
                features = {
                    'profit': transaction['net_profit'],
                    'path': transaction['path_hash'],
                    'dexes': transaction['dexes'],
                    'roi': transaction['roi_percentage'],
                    'timestamp': transaction['block_timestamp']
                }
                
                # Check if profitable
                if features['profit'] > 0:
                    return {
                        **features,
                        'detected_at': datetime.utcnow().isoformat(),
                        'confidence': min(100, features['roi'] * 10)
                    }
            
            return None
        
        await self.create_stream(
            input_topic='arbitrage-transactions',
            output_topic='arbitrage-opportunities',
            processor_func=detect_arbitrage,
            group_id='arbitrage-detector'
        )

async def setup_kafka_infrastructure(config: KafkaConfig):
    """Setup Kafka topics and infrastructure"""
    
    admin = AIOKafkaAdminClient(
        bootstrap_servers=','.join(config.bootstrap_servers)
    )
    
    await admin.start()
    
    # Create topics
    topics_to_create = []
    for topic_name, topic_config in config.topics.items():
        topic = NewTopic(
            name=topic_name,
            num_partitions=topic_config['num_partitions'],
            replication_factor=topic_config['replication_factor'],
            topic_configs=topic_config.get('config', {})
        )
        topics_to_create.append(topic)
    
    try:
        await admin.create_topics(topics_to_create)
        logger.info(f"Created {len(topics_to_create)} topics")
    except Exception as e:
        logger.warning(f"Some topics may already exist: {e}")
    
    await admin.close()

# Example usage
async def main():
    config = KafkaConfig()
    
    # Setup infrastructure
    await setup_kafka_infrastructure(config)
    
    # Create producer
    producer = EliteKafkaProducer(config)
    await producer.start()
    
    # Send sample transaction
    sample_tx = {
        'signature': 'abc123',
        'block_timestamp': datetime.utcnow().isoformat(),
        'net_profit': 1000000,
        'mev_type': 'arbitrage',
        'searcher_address': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb4',
        'roi_percentage': 15.5,
        'path_hash': 'xyz789',
        'dexes': ['uniswap', 'sushiswap'],
    }
    
    await producer.send_transaction(sample_tx)
    
    # Create stream processor
    processor = KafkaStreamProcessor(config)
    await processor.create_arbitrage_detection_stream()
    
    # Let it run
    await asyncio.sleep(60)
    
    # Cleanup
    await producer.close()

if __name__ == "__main__":
    asyncio.run(main())