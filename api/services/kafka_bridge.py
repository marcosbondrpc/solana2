"""
Kafka/Redpanda bridge for real-time message streaming
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
import logging
from google.protobuf.message import Message as ProtoMessage
from google.protobuf.json_format import MessageToDict
import importlib


logger = logging.getLogger(__name__)


class KafkaBridge:
    """Kafka/Redpanda consumer with backpressure handling"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "mev-api-consumer",
        topics: Optional[List[str]] = None
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or [
            "mev.opportunities",
            "mev.alerts.arbitrage",
            "mev.alerts.sandwich",
            "mev.metrics",
            "mev.decisions"
        ]
        
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None
        self.handlers: Dict[str, List[Callable]] = {}
        self.running = False
        self.backpressure_threshold = 1000
        self.message_buffer: asyncio.Queue = asyncio.Queue(maxsize=10000)
        
        # Proto message classes cache
        self.proto_classes = {}
        self._load_proto_classes()
    
    def _load_proto_classes(self):
        """Load protobuf message classes"""
        try:
            # Import generated protobuf modules
            from api.proto_gen import realtime_pb2, control_pb2
            
            self.proto_classes = {
                "mev.opportunities": realtime_pb2.MEVOpportunity,
                "mev.alerts.arbitrage": realtime_pb2.ArbitrageAlert,
                "mev.alerts.sandwich": realtime_pb2.SandwichAlert,
                "mev.metrics": realtime_pb2.SystemMetrics,
                "node.health": realtime_pb2.NodeHealth,
                "thompson.stats": realtime_pb2.ThompsonStats,
                "model.deployment": realtime_pb2.ModelDeployment,
                "job.progress": realtime_pb2.JobProgress
            }
        except ImportError:
            logger.warning("Protobuf classes not available, using raw bytes")
    
    async def start(self):
        """Start Kafka consumer and producer"""
        
        # Create consumer
        self.consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            value_deserializer=self._deserialize_message,
            max_poll_records=500,
            fetch_max_wait_ms=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        
        # Create producer for command responses
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=self._serialize_message,
            compression_type="snappy",
            max_batch_size=16384,
            linger_ms=10
        )
        
        try:
            await self.consumer.start()
            await self.producer.start()
            self.running = True
            
            # Start consumer task
            asyncio.create_task(self._consume_messages())
            asyncio.create_task(self._process_buffer())
            
            logger.info(f"Kafka bridge started, consuming topics: {self.topics}")
            
        except KafkaError as e:
            logger.error(f"Failed to start Kafka bridge: {e}")
            raise
    
    async def stop(self):
        """Stop Kafka consumer and producer"""
        self.running = False
        
        if self.consumer:
            await self.consumer.stop()
        
        if self.producer:
            await self.producer.stop()
        
        logger.info("Kafka bridge stopped")
    
    def _deserialize_message(self, data: bytes) -> Any:
        """Deserialize Kafka message"""
        try:
            # Try to decode as protobuf first
            # This is a simplified version - in production you'd need proper type detection
            return data
        except Exception:
            # Fallback to JSON
            try:
                return json.loads(data.decode())
            except Exception:
                return data
    
    def _serialize_message(self, data: Any) -> bytes:
        """Serialize message for Kafka"""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, ProtoMessage):
            return data.SerializeToString()
        else:
            return json.dumps(data).encode()
    
    async def _consume_messages(self):
        """Main consumer loop"""
        while self.running:
            try:
                # Fetch messages with timeout
                messages = await self.consumer.getmany(
                    timeout_ms=1000,
                    max_records=100
                )
                
                for topic_partition, records in messages.items():
                    topic = topic_partition.topic
                    
                    for record in records:
                        # Check backpressure
                        if self.message_buffer.qsize() > self.backpressure_threshold:
                            logger.warning(f"Backpressure threshold reached: {self.message_buffer.qsize()}")
                            await asyncio.sleep(0.1)
                        
                        # Add to buffer
                        message = {
                            "topic": topic,
                            "partition": record.partition,
                            "offset": record.offset,
                            "timestamp": record.timestamp,
                            "key": record.key,
                            "value": record.value,
                            "headers": dict(record.headers) if record.headers else {}
                        }
                        
                        try:
                            await self.message_buffer.put(message)
                        except asyncio.QueueFull:
                            logger.error("Message buffer full, dropping message")
                
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                await asyncio.sleep(1)
    
    async def _process_buffer(self):
        """Process messages from buffer"""
        while self.running:
            try:
                # Get message from buffer
                message = await asyncio.wait_for(
                    self.message_buffer.get(),
                    timeout=1.0
                )
                
                # Process message
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle individual message"""
        topic = message["topic"]
        value = message["value"]
        
        # Decode protobuf if available
        if topic in self.proto_classes and isinstance(value, bytes):
            try:
                proto_class = self.proto_classes[topic]
                proto_message = proto_class()
                proto_message.ParseFromString(value)
                value = MessageToDict(proto_message)
            except Exception as e:
                logger.debug(f"Failed to decode protobuf: {e}")
        
        # Call registered handlers
        if topic in self.handlers:
            for handler in self.handlers[topic]:
                try:
                    await handler(value, message)
                except Exception as e:
                    logger.error(f"Handler error for topic {topic}: {e}")
    
    def register_handler(self, topic: str, handler: Callable):
        """Register message handler for topic"""
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)
    
    async def send_command(self, topic: str, command: Dict[str, Any]) -> bool:
        """Send command message to Kafka"""
        if not self.producer:
            return False
        
        try:
            await self.producer.send(
                topic,
                value=command,
                timestamp_ms=int(datetime.now().timestamp() * 1000)
            )
            await self.producer.flush()
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    async def get_consumer_lag(self) -> Dict[str, int]:
        """Get consumer lag for all partitions"""
        if not self.consumer:
            return {}
        
        lag_info = {}
        
        try:
            # Get current positions
            partitions = self.consumer.assignment()
            
            for partition in partitions:
                # Get committed offset
                committed = await self.consumer.committed(partition)
                
                # Get latest offset
                end_offsets = await self.consumer.end_offsets([partition])
                latest = end_offsets.get(partition, 0)
                
                # Calculate lag
                lag = latest - (committed or 0)
                lag_info[f"{partition.topic}:{partition.partition}"] = lag
        
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
        
        return lag_info
    
    async def pause_consumption(self):
        """Pause message consumption (for backpressure)"""
        if self.consumer:
            self.consumer.pause(self.consumer.assignment())
    
    async def resume_consumption(self):
        """Resume message consumption"""
        if self.consumer:
            self.consumer.resume(self.consumer.assignment())
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "buffer_size": self.message_buffer.qsize(),
            "buffer_max_size": self.message_buffer.maxsize,
            "buffer_usage_pct": (self.message_buffer.qsize() / self.message_buffer.maxsize) * 100,
            "backpressure_active": self.message_buffer.qsize() > self.backpressure_threshold
        }


# Global Kafka bridge instance
kafka_bridge: Optional[KafkaBridge] = None


async def initialize_kafka_bridge(
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "mev-api-consumer",
    topics: Optional[List[str]] = None
):
    """Initialize global Kafka bridge"""
    global kafka_bridge
    
    kafka_bridge = KafkaBridge(
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        topics=topics
    )
    
    await kafka_bridge.start()


async def get_kafka_bridge() -> KafkaBridge:
    """Get global Kafka bridge instance"""
    if kafka_bridge is None:
        await initialize_kafka_bridge()
    return kafka_bridge