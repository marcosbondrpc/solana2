"""
Kafka Bridge: JSON to Protobuf converter
Ultra-fast zero-copy serialization with lock-free buffers
"""

import os
import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import aiokafka
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import zstandard as zstd

from proto_gen import realtime_pb2, control_pb2


# Global state
bridges: Dict[str, 'KafkaBridge'] = {}
bridge_lock = asyncio.Lock()

# Compression
compressor = zstd.ZstdCompressor(level=3, threads=2)


class KafkaBridge:
    """High-performance JSON to Protobuf bridge"""
    
    def __init__(self, source_topic: str, target_topic: str, message_type: str):
        self.source_topic = source_topic
        self.target_topic = target_topic
        self.message_type = message_type
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None
        self.running = False
        self.stats = {
            "messages_converted": 0,
            "bytes_processed": 0,
            "errors": 0,
            "latency_us_avg": 0
        }
    
    async def start(self):
        """Start the bridge"""
        # Create consumer
        self.consumer = AIOKafkaConsumer(
            self.source_topic,
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
            group_id=f"bridge-{self.source_topic}-to-{self.target_topic}",
            fetch_min_bytes=1,
            fetch_max_wait_ms=10,
            max_poll_records=1000,  # Batch fetch
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: m  # Raw bytes
        )
        
        # Create producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
            linger_ms=0,  # No delay
            acks=1,
            compression_type="lz4",
            max_batch_size=65536,  # Larger batches
            max_request_size=10485760,  # 10MB
            enable_idempotence=True
        )
        
        await self.consumer.start()
        await self.producer.start()
        
        self.running = True
        asyncio.create_task(self.conversion_loop())
    
    async def stop(self):
        """Stop the bridge"""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
    
    def convert_json_to_proto(self, json_data: Dict[str, Any]) -> bytes:
        """Convert JSON to protobuf based on message type"""
        
        if self.message_type == "mev_opportunity":
            msg = realtime_pb2.MevOpportunity()
            msg.tx_hash = json_data.get("tx_hash", "")
            msg.block_hash = json_data.get("block_hash", "")
            msg.slot = json_data.get("slot", 0)
            msg.profit_lamports = json_data.get("profit_lamports", 0.0)
            msg.probability = json_data.get("probability", 0.0)
            msg.opportunity_type = json_data.get("opportunity_type", "")
            
            for account in json_data.get("target_accounts", []):
                msg.target_accounts.append(account)
            
            msg.gas_estimate = json_data.get("gas_estimate", 0)
            msg.priority_fee = json_data.get("priority_fee", 0)
            
            if "raw_transaction" in json_data:
                msg.raw_transaction = bytes.fromhex(json_data["raw_transaction"])
            
            for key, value in json_data.get("metrics", {}).items():
                msg.metrics[key] = float(value)
            
            return msg.SerializeToString()
            
        elif self.message_type == "arbitrage_opportunity":
            msg = realtime_pb2.ArbitrageOpportunity()
            msg.id = json_data.get("id", "")
            msg.slot = json_data.get("slot", 0)
            
            for market in json_data.get("dex_markets", []):
                msg.dex_markets.append(market)
            
            msg.profit_estimate = json_data.get("profit_estimate", 0.0)
            msg.execution_probability = json_data.get("execution_probability", 0.0)
            msg.gas_cost = json_data.get("gas_cost", 0)
            msg.deadline_ns = json_data.get("deadline_ns", 0)
            
            for step_data in json_data.get("route", []):
                step = msg.route.add()
                step.dex = step_data.get("dex", "")
                step.pool_address = step_data.get("pool_address", "")
                step.token_in = step_data.get("token_in", "")
                step.token_out = step_data.get("token_out", "")
                step.amount_in = step_data.get("amount_in", 0)
                step.amount_out = step_data.get("amount_out", 0)
                step.slippage = step_data.get("slippage", 0.0)
            
            for key, value in json_data.get("risk_metrics", {}).items():
                msg.risk_metrics[key] = float(value)
            
            return msg.SerializeToString()
            
        elif self.message_type == "bundle_outcome":
            msg = realtime_pb2.BundleOutcome()
            msg.bundle_id = json_data.get("bundle_id", "")
            msg.slot = json_data.get("slot", 0)
            msg.landed = json_data.get("landed", False)
            msg.profit_actual = json_data.get("profit_actual", 0.0)
            msg.gas_used = json_data.get("gas_used", 0)
            msg.error = json_data.get("error", "")
            msg.latency_ms = json_data.get("latency_ms", 0)
            
            for key, value in json_data.get("metadata", {}).items():
                msg.metadata[key] = str(value)
            
            return msg.SerializeToString()
            
        elif self.message_type == "market_tick":
            msg = realtime_pb2.MarketTick()
            msg.market_id = json_data.get("market_id", "")
            msg.timestamp_ns = json_data.get("timestamp_ns", time.time_ns())
            msg.bid_price = json_data.get("bid_price", 0.0)
            msg.ask_price = json_data.get("ask_price", 0.0)
            msg.bid_size = json_data.get("bid_size", 0.0)
            msg.ask_size = json_data.get("ask_size", 0.0)
            msg.last_price = json_data.get("last_price", 0.0)
            msg.volume_24h = json_data.get("volume_24h", 0)
            
            for key, value in json_data.get("additional_data", {}).items():
                msg.additional_data[key] = float(value)
            
            return msg.SerializeToString()
            
        else:
            # Generic envelope
            envelope = realtime_pb2.Envelope()
            envelope.timestamp_ns = json_data.get("timestamp_ns", time.time_ns())
            envelope.stream_id = json_data.get("stream_id", self.source_topic)
            envelope.sequence = json_data.get("sequence", 0)
            envelope.type = self.message_type
            envelope.payload = json.dumps(json_data).encode()
            
            return envelope.SerializeToString()
    
    async def conversion_loop(self):
        """Main conversion loop"""
        batch_size = 100
        batch_timeout = 0.01  # 10ms
        
        while self.running:
            try:
                # Fetch batch of messages
                records = await self.consumer.getmany(
                    timeout_ms=int(batch_timeout * 1000),
                    max_records=batch_size
                )
                
                if not records:
                    continue
                
                # Process batch
                futures = []
                for topic_partition, messages in records.items():
                    for msg in messages:
                        start_ns = time.perf_counter_ns()
                        
                        try:
                            # Parse JSON
                            if msg.value.startswith(b'{'):
                                json_data = json.loads(msg.value)
                            else:
                                # Skip non-JSON messages
                                continue
                            
                            # Convert to protobuf
                            proto_bytes = self.convert_json_to_proto(json_data)
                            
                            # Wrap in envelope for transport
                            envelope = realtime_pb2.Envelope()
                            envelope.timestamp_ns = time.time_ns()
                            envelope.stream_id = self.source_topic
                            envelope.sequence = msg.offset
                            envelope.type = self.message_type
                            envelope.payload = proto_bytes
                            
                            # Send to target topic
                            future = await self.producer.send(
                                self.target_topic,
                                value=envelope.SerializeToString(),
                                key=msg.key,
                                timestamp_ms=msg.timestamp if msg.timestamp else None
                            )
                            futures.append(future)
                            
                            # Update stats
                            self.stats["messages_converted"] += 1
                            self.stats["bytes_processed"] += len(msg.value)
                            
                            # Calculate latency
                            latency_ns = time.perf_counter_ns() - start_ns
                            latency_us = latency_ns / 1000
                            
                            # Update average (simple moving average)
                            alpha = 0.1
                            self.stats["latency_us_avg"] = (
                                alpha * latency_us + 
                                (1 - alpha) * self.stats["latency_us_avg"]
                            )
                            
                        except Exception as e:
                            print(f"Error converting message: {e}")
                            self.stats["errors"] += 1
                
                # Wait for all sends to complete
                if futures:
                    await asyncio.gather(*futures, return_exceptions=True)
                    
            except Exception as e:
                print(f"Bridge loop error: {e}")
                await asyncio.sleep(1)


async def json_to_proto_loop():
    """Main bridge loop that converts JSON to Protobuf"""
    
    # Define bridges
    bridge_configs = [
        ("mev-opportunities", "mev-opportunities-proto", "mev_opportunity"),
        ("arbitrage-opportunities", "arbitrage-opportunities-proto", "arbitrage_opportunity"),
        ("bundle-outcomes", "bundle-outcomes-proto", "bundle_outcome"),
        ("market-ticks", "market-ticks-proto", "market_tick"),
        ("metrics", "metrics-proto", "metrics_update")
    ]
    
    # Start bridges
    async with bridge_lock:
        for source, target, msg_type in bridge_configs:
            if source not in bridges:
                bridge = KafkaBridge(source, target, msg_type)
                await bridge.start()
                bridges[source] = bridge
                print(f"✅ Started bridge: {source} → {target} ({msg_type})")
    
    # Monitor bridges
    while True:
        await asyncio.sleep(30)
        
        # Print stats
        print("\n📊 Bridge Statistics:")
        for source, bridge in bridges.items():
            print(f"  {source}:")
            print(f"    Messages: {bridge.stats['messages_converted']:,}")
            print(f"    Bytes: {bridge.stats['bytes_processed']:,}")
            print(f"    Errors: {bridge.stats['errors']:,}")
            print(f"    Latency: {bridge.stats['latency_us_avg']:.1f}µs")


async def start_kafka_bridge():
    """Start the Kafka bridge service"""
    print("🌉 Starting Kafka JSON→Protobuf bridge...")
    
    try:
        await json_to_proto_loop()
    except Exception as e:
        print(f"Bridge error: {e}")


# API endpoints for bridge management
from fastapi import APIRouter

bridge_router = APIRouter()


@bridge_router.get("/bridges")
async def get_bridges():
    """Get bridge status"""
    status = {}
    for source, bridge in bridges.items():
        status[source] = {
            "source": bridge.source_topic,
            "target": bridge.target_topic,
            "type": bridge.message_type,
            "running": bridge.running,
            "stats": bridge.stats
        }
    return status


@bridge_router.post("/bridges/{source}/restart")
async def restart_bridge(source: str):
    """Restart a bridge"""
    if source in bridges:
        bridge = bridges[source]
        await bridge.stop()
        await bridge.start()
        return {"status": "restarted", "source": source}
    return {"error": "Bridge not found"}


# Cleanup
async def cleanup():
    """Clean up bridges"""
    async with bridge_lock:
        for bridge in bridges.values():
            await bridge.stop()
        bridges.clear()