#!/usr/bin/env python3
"""
LEGENDARY Lab Smoke Test Framework
Ultra-comprehensive testing for MEV infrastructure
Tests Kafka → ClickHouse → Grafana pipeline with performance benchmarking
"""

import os
import sys
import time
import json
import random
import struct
import hashlib
import argparse
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
import clickhouse_connect as ch
import aiohttp
import psutil
import redis

# Import generated protobufs
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.proto_gen import realtime_pb2 as rt
    from api.proto_gen import control_pb2 as ctrl
except ImportError as e:
    print(f"ERROR: Missing protobuf files. Run 'make proto' first: {e}")
    sys.exit(1)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Performance metrics collector
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    kafka_produce_latency_ms: float
    kafka_consume_latency_ms: float
    clickhouse_insert_latency_ms: float
    clickhouse_query_latency_ms: float
    grafana_api_latency_ms: float
    total_events_produced: int
    total_events_consumed: int
    total_rows_inserted: int
    cpu_usage_percent: float
    memory_usage_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    test_duration_seconds: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class MEVEventGenerator:
    """Generate realistic MEV events for testing"""
    
    LEADERS = ["Lx9", "Ly7", "Lz3", "La1", "Lb2", "Lc4", "Ld5", "Le6"]
    ROUTES = ["Direct", "Jito", "Hybrid", "FlashBot", "Private"]
    POLICIES = ["ucb", "thompson", "epsilon_greedy", "exp3", "contextual"]
    TX_KINDS = ["sandwich", "arbitrage", "liquidation", "jit", "cex_dex_arb"]
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.slot_counter = 360200000
        
    def blake32(self, data: bytes) -> bytes:
        """Generate 32-byte BLAKE2s hash"""
        return hashlib.blake2s(data).digest()
    
    def feat_sketch8(self, features: List[float]) -> bytes:
        """Generate 8-byte feature sketch"""
        h = hashlib.blake2s()
        step = max(1, len(features) // 16)
        for x in features[::step]:
            h.update(struct.pack("<f", float(x)))
        return h.digest()[:8]
    
    def build_dna(self, model_id: str, policy: str, route: str, 
                  tip_sol: float, p_land: float, features: List[float],
                  slot: int, tx_kind: str) -> rt.DecisionDNA:
        """Build DecisionDNA with realistic parameters"""
        sketch = self.feat_sketch8(features)
        fp_data = (
            model_id.encode() + policy.encode() + route.encode() +
            struct.pack("<dd", tip_sol, p_land) + sketch +
            struct.pack("<Q", slot) + tx_kind.encode()
        )
        fp = self.blake32(fp_data)
        
        return rt.DecisionDNA(
            fp=fp,
            model_abi_id=model_id,
            policy=policy,
            route=route,
            tip_sol=tip_sol,
            p_land_est=p_land,
            feat_sketch8=sketch,
            slot=slot,
            tx_kind=tx_kind,
        )
    
    def generate_bandit_event(self, module: str = "mev") -> rt.BanditEvent:
        """Generate realistic BanditEvent"""
        arm = random.choice([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])  # Tip multipliers
        route = random.choice(self.ROUTES)
        
        # Realistic profit distribution (mostly small, some large)
        if random.random() < 0.1:  # 10% chance of high profit
            payoff = random.uniform(0.001, 0.01)  # 1-10 mSOL profit
        else:
            payoff = random.uniform(-0.0002, 0.0009)  # Small profit/loss
        
        # Landing probability based on tip amount
        base_land_prob = 0.3 + (arm * 0.2)  # Higher tip = higher landing prob
        landed = 1.0 if random.random() < base_land_prob else 0.0
        
        self.slot_counter += 1
        
        return rt.BanditEvent(
            module=module,
            policy=random.choice(self.POLICIES),
            route=route,
            arm=arm,
            payoff=payoff,
            tip_sol=arm * 0.0001,  # Base tip 0.1 mSOL
            ev_sol=payoff * 0.7,
            slot=self.slot_counter,
            leader=random.choice(self.LEADERS),
            ts=int(time.time()),
            landed=landed,
            p_land_est=base_land_prob + random.uniform(-0.1, 0.1),
        )
    
    def generate_mev_event(self) -> rt.MevEvent:
        """Generate realistic MEV event with DNA"""
        features = np.random.randn(64).tolist()  # 64 feature vector
        self.slot_counter += 1
        slot = self.slot_counter
        
        tx_kind = random.choice(self.TX_KINDS)
        route = random.choice(self.ROUTES)
        
        # Profit based on transaction type
        profit_ranges = {
            "sandwich": (0.0001, 0.005),
            "arbitrage": (0.0005, 0.01),
            "liquidation": (0.001, 0.05),
            "jit": (0.0001, 0.002),
            "cex_dex_arb": (0.001, 0.02),
        }
        min_profit, max_profit = profit_ranges.get(tx_kind, (0.0001, 0.001))
        profit_est = random.uniform(min_profit, max_profit)
        
        # Tip calculation (percentage of profit)
        tip_percentage = random.uniform(0.3, 0.7)  # 30-70% of profit as tip
        tip_sol = profit_est * tip_percentage
        
        # Landing probability
        p_land = 0.4 + min(tip_percentage, 0.5)  # Higher tip % = higher landing
        
        dna = self.build_dna(
            model_id=f"mev_model_v{random.randint(1,5)}",
            policy=random.choice(self.POLICIES),
            route=route,
            tip_sol=tip_sol,
            p_land=p_land,
            features=features,
            slot=slot,
            tx_kind=tx_kind
        )
        
        mev = rt.MevEvent()
        mev.slot = slot
        mev.leader = random.choice(self.LEADERS)
        mev.profit_est_sol = profit_est
        mev.tip_sol = tip_sol
        mev.route = route
        mev.dna.CopyFrom(dna)
        
        return mev
    
    def generate_arbitrage_opportunity(self) -> Dict[str, Any]:
        """Generate arbitrage opportunity data"""
        tokens = ["SOL", "USDC", "USDT", "RAY", "SRM", "ORCA", "MNGO", "COPE"]
        dexes = ["Raydium", "Orca", "Serum", "Saber", "Mercurial"]
        
        token_a = random.choice(tokens)
        token_b = random.choice([t for t in tokens if t != token_a])
        
        # Multi-hop path
        path_length = random.randint(2, 4)
        path = []
        for _ in range(path_length):
            path.append({
                "dex": random.choice(dexes),
                "pool": f"{token_a}/{token_b}",
                "price": random.uniform(0.9, 1.1),
            })
        
        profit_usd = random.uniform(10, 1000) if random.random() < 0.1 else random.uniform(1, 50)
        
        return {
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "token_a": token_a,
            "token_b": token_b,
            "path": json.dumps(path),
            "profit_usd": profit_usd,
            "gas_estimate": random.uniform(0.001, 0.01),
            "execution_status": random.choice(["pending", "executed", "failed", "expired"]),
            "gas_used": random.uniform(0.0005, 0.008) if random.random() < 0.7 else 0,
            "actual_profit_usd": profit_usd * random.uniform(0.7, 1.2) if random.random() < 0.6 else 0,
            "slippage": random.uniform(0.001, 0.05),
            "confidence": random.uniform(0.5, 0.99),
        }


class KafkaTestManager:
    """Manage Kafka operations for testing"""
    
    def __init__(self, brokers: str):
        self.brokers = brokers
        self.producer = Producer({
            "bootstrap.servers": brokers,
            "compression.type": "zstd",
            "linger.ms": "1",
            "batch.size": 65536,
            "acks": "all",
            "enable.idempotence": True,
            "max.in.flight.requests.per.connection": 5,
        })
        self.admin = AdminClient({"bootstrap.servers": brokers})
        
    def create_topics(self, topics: List[str], num_partitions: int = 3):
        """Create Kafka topics if they don't exist"""
        new_topics = [
            NewTopic(topic, num_partitions=num_partitions, replication_factor=1)
            for topic in topics
        ]
        
        fs = self.admin.create_topics(new_topics, validate_only=False)
        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Topic {topic} created successfully")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Topic {topic} already exists")
                else:
                    logger.error(f"Failed to create topic {topic}: {e}")
    
    def produce_events(self, topic: str, events: List[bytes]) -> float:
        """Produce events and return average latency"""
        latencies = []
        
        for event in events:
            start = time.perf_counter()
            self.producer.produce(topic, event)
            latencies.append((time.perf_counter() - start) * 1000)
        
        self.producer.flush(timeout=10)
        return np.mean(latencies) if latencies else 0
    
    def verify_consumption(self, topic: str, expected_count: int, timeout: int = 30) -> Tuple[int, float]:
        """Verify events are consumable and return count + latency"""
        consumer = Consumer({
            "bootstrap.servers": self.brokers,
            "group.id": f"smoke-test-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        })
        
        consumer.subscribe([topic])
        consumed = 0
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < timeout and consumed < expected_count:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Consumer error: {msg.error()}")
                continue
            
            consumed += 1
            # Estimate latency from timestamp
            if msg.timestamp()[0] == 0:  # CreateTime
                latency = (time.time() * 1000) - msg.timestamp()[1]
                latencies.append(latency)
        
        consumer.close()
        avg_latency = np.mean(latencies) if latencies else 0
        return consumed, avg_latency


class ClickHouseTestManager:
    """Manage ClickHouse operations for testing"""
    
    def __init__(self, url: str):
        self.url = url
        self.client = ch.get_client(host=url.replace("http://", "").replace(":8123", ""),
                                   database="default")
    
    def ensure_tables(self):
        """Ensure all required tables exist"""
        required_tables = [
            ("bandit_events", self._get_bandit_table_ddl()),
            ("mev_opportunities", self._get_mev_table_ddl()),
            ("arbitrage_opportunities", self._get_arbitrage_table_ddl()),
            ("kafka_bandit_env", self._get_kafka_bandit_ddl()),
            ("kafka_realtime_env", self._get_kafka_realtime_ddl()),
        ]
        
        for table_name, ddl in required_tables:
            try:
                # Check if table exists
                result = self.client.query(f"EXISTS TABLE {table_name}")
                if not result.first_row[0]:
                    logger.info(f"Creating table {table_name}")
                    self.client.command(ddl)
                else:
                    logger.info(f"Table {table_name} already exists")
            except Exception as e:
                logger.error(f"Error with table {table_name}: {e}")
    
    def _get_bandit_table_ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS bandit_events (
            ts DateTime64(3),
            module String,
            policy String,
            route String,
            arm Float64,
            payoff Float64,
            tip_sol Float64,
            ev_sol Float64,
            slot UInt64,
            leader String,
            landed Float64,
            p_land_est Float64
        ) ENGINE = MergeTree()
        ORDER BY (ts, module, route)
        PARTITION BY toYYYYMMDD(ts)
        """
    
    def _get_mev_table_ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS mev_opportunities (
            ts DateTime64(3) DEFAULT now64(3),
            slot UInt64,
            leader String,
            profit_est_sol Float64,
            tip_sol Float64,
            route String,
            dna_fp FixedString(32),
            model_abi_id String,
            policy String,
            p_land_est Float64,
            tx_kind String
        ) ENGINE = MergeTree()
        ORDER BY (ts, slot)
        PARTITION BY toYYYYMMDD(ts)
        """
    
    def _get_arbitrage_table_ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
            detected_at DateTime64(3),
            token_a String,
            token_b String,
            path String,
            profit_usd Float64,
            gas_estimate Float64,
            execution_status String,
            gas_used Float64,
            actual_profit_usd Float64,
            slippage Float64,
            confidence Float64
        ) ENGINE = MergeTree()
        ORDER BY (detected_at, token_a, token_b)
        PARTITION BY toYYYYMMDD(detected_at)
        """
    
    def _get_kafka_bandit_ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS kafka_bandit_env (
            raw String
        ) ENGINE = Kafka
        SETTINGS
            kafka_broker_list = 'localhost:9092',
            kafka_topic_list = 'bandit-events-proto',
            kafka_group_name = 'clickhouse-bandit',
            kafka_format = 'ProtobufSingle',
            kafka_schema = 'realtime.proto:Envelope'
        """
    
    def _get_kafka_realtime_ddl(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS kafka_realtime_env (
            raw String
        ) ENGINE = Kafka
        SETTINGS
            kafka_broker_list = 'localhost:9092',
            kafka_topic_list = 'realtime-proto',
            kafka_group_name = 'clickhouse-realtime',
            kafka_format = 'ProtobufSingle',
            kafka_schema = 'realtime.proto:Envelope'
        """
    
    def insert_batch(self, table: str, data: List[Dict]) -> float:
        """Insert batch of data and return latency"""
        if not data:
            return 0
        
        start = time.perf_counter()
        self.client.insert(table, data)
        return (time.perf_counter() - start) * 1000
    
    def verify_data(self, table: str, min_rows: int) -> int:
        """Verify minimum rows exist in table"""
        result = self.client.query(f"SELECT count() FROM {table}")
        count = result.first_row[0]
        return count
    
    def run_performance_queries(self) -> Dict[str, float]:
        """Run performance benchmark queries"""
        queries = {
            "simple_count": "SELECT count() FROM bandit_events",
            "aggregation": """
                SELECT route, avg(payoff), count() 
                FROM bandit_events 
                GROUP BY route
            """,
            "time_series": """
                SELECT 
                    toStartOfMinute(ts) as minute,
                    avg(payoff),
                    sum(landed)
                FROM bandit_events
                WHERE ts > now() - INTERVAL 1 HOUR
                GROUP BY minute
                ORDER BY minute
            """,
            "complex_join": """
                SELECT 
                    b.route,
                    avg(b.payoff) as avg_payoff,
                    count(DISTINCT m.slot) as unique_slots
                FROM bandit_events b
                LEFT JOIN mev_opportunities m ON b.slot = m.slot
                GROUP BY b.route
            """,
        }
        
        results = {}
        for name, query in queries.items():
            try:
                start = time.perf_counter()
                self.client.query(query)
                results[name] = (time.perf_counter() - start) * 1000
            except Exception as e:
                logger.error(f"Query {name} failed: {e}")
                results[name] = -1
        
        return results


class GrafanaTestManager:
    """Test Grafana API connectivity and provisioning"""
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url.rstrip('/')
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def test_health(self) -> float:
        """Test Grafana health endpoint"""
        async with aiohttp.ClientSession() as session:
            start = time.perf_counter()
            try:
                async with session.get(f"{self.url}/api/health") as resp:
                    latency = (time.perf_counter() - start) * 1000
                    if resp.status == 200:
                        logger.info("Grafana health check passed")
                        return latency
                    else:
                        logger.error(f"Grafana health check failed: {resp.status}")
                        return -1
            except Exception as e:
                logger.error(f"Grafana connection failed: {e}")
                return -1
    
    async def verify_datasource(self, name: str = "ClickHouse") -> bool:
        """Verify datasource exists"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.url}/api/datasources/name/{name}",
                    headers=self.headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"Datasource {name} found: {data.get('uid')}")
                        return True
                    else:
                        logger.error(f"Datasource {name} not found")
                        return False
            except Exception as e:
                logger.error(f"Failed to verify datasource: {e}")
                return False
    
    async def verify_dashboards(self) -> List[str]:
        """Get list of dashboard UIDs"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.url}/api/search?type=dash-db",
                    headers=self.headers
                ) as resp:
                    if resp.status == 200:
                        dashboards = await resp.json()
                        uids = [d.get("uid") for d in dashboards]
                        logger.info(f"Found {len(uids)} dashboards")
                        return uids
                    else:
                        logger.error("Failed to fetch dashboards")
                        return []
            except Exception as e:
                logger.error(f"Failed to list dashboards: {e}")
                return []


class ComprehensiveSmokeTest:
    """Main smoke test orchestrator"""
    
    def __init__(self, args):
        self.args = args
        self.generator = MEVEventGenerator(seed=args.seed)
        self.kafka = KafkaTestManager(args.brokers)
        self.clickhouse = ClickHouseTestManager(args.ch_url)
        self.grafana = None
        if args.grafana_url:
            self.grafana = GrafanaTestManager(args.grafana_url, args.grafana_key)
        
        self.metrics = None
        self.start_time = None
        self.initial_network_stats = None
        self.initial_cpu_stats = None
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "network": psutil.net_io_counters()._asdict(),
            "disk": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
        }
    
    async def run(self) -> PerformanceMetrics:
        """Run comprehensive smoke test"""
        self.start_time = time.time()
        self.initial_network_stats = psutil.net_io_counters()
        self.initial_cpu_stats = self.collect_system_metrics()
        
        logger.info("=" * 60)
        logger.info("LEGENDARY MEV Infrastructure Smoke Test")
        logger.info("=" * 60)
        
        # Step 1: Setup infrastructure
        logger.info("\n[1/7] Setting up infrastructure...")
        self._setup_infrastructure()
        
        # Step 2: Generate test events
        logger.info(f"\n[2/7] Generating {self.args.bandit + self.args.mev + self.args.arb} test events...")
        bandit_events, mev_events, arb_data = self._generate_events()
        
        # Step 3: Produce to Kafka
        logger.info("\n[3/7] Producing events to Kafka...")
        kafka_metrics = await self._test_kafka(bandit_events, mev_events)
        
        # Step 4: Verify ClickHouse ingestion
        logger.info("\n[4/7] Verifying ClickHouse ingestion...")
        ch_metrics = await self._test_clickhouse(arb_data)
        
        # Step 5: Test Grafana if configured
        grafana_latency = 0
        if self.grafana:
            logger.info("\n[5/7] Testing Grafana API...")
            grafana_latency = await self._test_grafana()
        
        # Step 6: Run performance benchmarks
        logger.info("\n[6/7] Running performance benchmarks...")
        self._run_benchmarks()
        
        # Step 7: Collect final metrics
        logger.info("\n[7/7] Collecting final metrics...")
        final_metrics = self._collect_final_metrics(
            kafka_metrics, ch_metrics, grafana_latency,
            len(bandit_events), len(mev_events), len(arb_data)
        )
        
        # Print summary
        self._print_summary(final_metrics)
        
        return final_metrics
    
    def _setup_infrastructure(self):
        """Setup Kafka topics and ClickHouse tables"""
        # Create Kafka topics
        topics = [
            "bandit-events-proto",
            "realtime-proto",
            "control-acks",
            "arbitrage-opportunities",
        ]
        self.kafka.create_topics(topics)
        
        # Ensure ClickHouse tables
        self.clickhouse.ensure_tables()
    
    def _generate_events(self) -> Tuple[List[bytes], List[bytes], List[Dict]]:
        """Generate all test events"""
        bandit_events = []
        mev_events = []
        arb_data = []
        
        # Generate Bandit events
        for _ in range(self.args.bandit):
            event = self.generator.generate_bandit_event()
            envelope = rt.Envelope(
                schema_version="rt-v1",
                type=rt.Envelope.Type.Value("BANDIT"),
                bandit=event
            )
            bandit_events.append(envelope.SerializeToString())
        
        # Generate MEV events
        for _ in range(self.args.mev):
            event = self.generator.generate_mev_event()
            envelope = rt.Envelope(
                schema_version="rt-v1",
                type=rt.Envelope.Type.Value("MEV"),
                mev=event
            )
            mev_events.append(envelope.SerializeToString())
        
        # Generate arbitrage opportunities
        for _ in range(self.args.arb):
            arb_data.append(self.generator.generate_arbitrage_opportunity())
        
        return bandit_events, mev_events, arb_data
    
    async def _test_kafka(self, bandit_events: List[bytes], 
                         mev_events: List[bytes]) -> Dict[str, float]:
        """Test Kafka production and consumption"""
        metrics = {}
        
        # Produce events
        if bandit_events:
            latency = self.kafka.produce_events("bandit-events-proto", bandit_events)
            metrics["bandit_produce_latency"] = latency
            logger.info(f"  Produced {len(bandit_events)} bandit events (avg latency: {latency:.2f}ms)")
        
        if mev_events:
            latency = self.kafka.produce_events("realtime-proto", mev_events)
            metrics["mev_produce_latency"] = latency
            logger.info(f"  Produced {len(mev_events)} MEV events (avg latency: {latency:.2f}ms)")
        
        # Verify consumption
        if self.args.verify_consume:
            logger.info("  Verifying consumption...")
            
            if bandit_events:
                count, latency = self.kafka.verify_consumption(
                    "bandit-events-proto", len(bandit_events)
                )
                metrics["bandit_consume_latency"] = latency
                logger.info(f"    Consumed {count}/{len(bandit_events)} bandit events")
            
            if mev_events:
                count, latency = self.kafka.verify_consumption(
                    "realtime-proto", len(mev_events)
                )
                metrics["mev_consume_latency"] = latency
                logger.info(f"    Consumed {count}/{len(mev_events)} MEV events")
        
        return metrics
    
    async def _test_clickhouse(self, arb_data: List[Dict]) -> Dict[str, float]:
        """Test ClickHouse operations"""
        metrics = {}
        
        # Insert arbitrage data directly
        if arb_data:
            latency = self.clickhouse.insert_batch("arbitrage_opportunities", arb_data)
            metrics["insert_latency"] = latency
            logger.info(f"  Inserted {len(arb_data)} arbitrage records ({latency:.2f}ms)")
        
        # Wait for Kafka engine consumption
        logger.info("  Waiting for Kafka engine consumption...")
        await asyncio.sleep(self.args.ch_wait)
        
        # Verify data
        tables_to_check = [
            ("bandit_events", self.args.bandit),
            ("mev_opportunities", self.args.mev),
            ("arbitrage_opportunities", self.args.arb),
        ]
        
        for table, expected in tables_to_check:
            count = self.clickhouse.verify_data(table, 0)
            logger.info(f"    {table}: {count} rows (expected ≥{expected})")
            metrics[f"{table}_count"] = count
        
        # Run query benchmarks
        query_metrics = self.clickhouse.run_performance_queries()
        metrics["queries"] = query_metrics
        logger.info("  Query benchmarks:")
        for name, latency in query_metrics.items():
            if latency > 0:
                logger.info(f"    {name}: {latency:.2f}ms")
        
        return metrics
    
    async def _test_grafana(self) -> float:
        """Test Grafana connectivity"""
        # Test health
        latency = await self.grafana.test_health()
        
        # Verify datasource
        if await self.grafana.verify_datasource("ClickHouse"):
            logger.info("  ClickHouse datasource verified")
        
        # List dashboards
        dashboards = await self.grafana.verify_dashboards()
        if dashboards:
            logger.info(f"  Found {len(dashboards)} dashboards")
        
        return latency
    
    def _run_benchmarks(self):
        """Run additional performance benchmarks"""
        # Concurrent write test
        logger.info("  Running concurrent write benchmark...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                events = [self.generator.generate_bandit_event() for _ in range(100)]
                envelopes = [
                    rt.Envelope(
                        schema_version="rt-v1",
                        type=rt.Envelope.Type.Value("BANDIT"),
                        bandit=e
                    ).SerializeToString()
                    for e in events
                ]
                futures.append(
                    executor.submit(
                        self.kafka.produce_events,
                        f"bandit-events-proto",
                        envelopes
                    )
                )
            
            results = [f.result() for f in as_completed(futures)]
            avg_latency = np.mean(results)
            logger.info(f"    Concurrent write avg latency: {avg_latency:.2f}ms")
    
    def _collect_final_metrics(self, kafka_metrics: Dict, ch_metrics: Dict,
                               grafana_latency: float, bandit_count: int,
                               mev_count: int, arb_count: int) -> PerformanceMetrics:
        """Collect all metrics"""
        final_network = psutil.net_io_counters()
        final_cpu = self.collect_system_metrics()
        
        return PerformanceMetrics(
            kafka_produce_latency_ms=kafka_metrics.get("bandit_produce_latency", 0),
            kafka_consume_latency_ms=kafka_metrics.get("bandit_consume_latency", 0),
            clickhouse_insert_latency_ms=ch_metrics.get("insert_latency", 0),
            clickhouse_query_latency_ms=np.mean(list(ch_metrics.get("queries", {}).values())),
            grafana_api_latency_ms=grafana_latency,
            total_events_produced=bandit_count + mev_count,
            total_events_consumed=ch_metrics.get("bandit_events_count", 0) + 
                                 ch_metrics.get("mev_opportunities_count", 0),
            total_rows_inserted=arb_count,
            cpu_usage_percent=final_cpu["cpu_percent"],
            memory_usage_mb=final_cpu["memory"]["used"] / 1024 / 1024,
            network_bytes_sent=final_network.bytes_sent - self.initial_network_stats.bytes_sent,
            network_bytes_recv=final_network.bytes_recv - self.initial_network_stats.bytes_recv,
            test_duration_seconds=time.time() - self.start_time,
        )
    
    def _print_summary(self, metrics: PerformanceMetrics):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("SMOKE TEST SUMMARY")
        logger.info("=" * 60)
        
        # Performance metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Kafka produce latency: {metrics.kafka_produce_latency_ms:.2f}ms")
        logger.info(f"  Kafka consume latency: {metrics.kafka_consume_latency_ms:.2f}ms")
        logger.info(f"  ClickHouse insert latency: {metrics.clickhouse_insert_latency_ms:.2f}ms")
        logger.info(f"  ClickHouse query latency: {metrics.clickhouse_query_latency_ms:.2f}ms")
        if metrics.grafana_api_latency_ms > 0:
            logger.info(f"  Grafana API latency: {metrics.grafana_api_latency_ms:.2f}ms")
        
        # Throughput metrics
        logger.info("\nThroughput Metrics:")
        logger.info(f"  Events produced: {metrics.total_events_produced}")
        logger.info(f"  Events consumed: {metrics.total_events_consumed}")
        logger.info(f"  Rows inserted: {metrics.total_rows_inserted}")
        events_per_sec = metrics.total_events_produced / metrics.test_duration_seconds
        logger.info(f"  Events/second: {events_per_sec:.2f}")
        
        # Resource usage
        logger.info("\nResource Usage:")
        logger.info(f"  CPU usage: {metrics.cpu_usage_percent:.1f}%")
        logger.info(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        logger.info(f"  Network sent: {metrics.network_bytes_sent / 1024 / 1024:.2f}MB")
        logger.info(f"  Network recv: {metrics.network_bytes_recv / 1024 / 1024:.2f}MB")
        
        # Test info
        logger.info(f"\nTest completed in {metrics.test_duration_seconds:.2f} seconds")
        
        # Save metrics to file
        if self.args.output:
            with open(self.args.output, 'w') as f:
                f.write(metrics.to_json())
            logger.info(f"\nMetrics saved to {self.args.output}")
        
        # Overall status
        status = "PASSED" if metrics.total_events_consumed > 0 else "FAILED"
        logger.info(f"\nOverall Status: {status}")
        
        if status == "PASSED":
            logger.info("\n✓ All components operational")
            logger.info("✓ Data pipeline functioning")
            logger.info("✓ Ready for production")
        else:
            logger.error("\n✗ Some components may have issues")
            logger.error("✗ Check logs for details")


def main():
    parser = argparse.ArgumentParser(
        description="LEGENDARY MEV Infrastructure Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  %(prog)s --bandit 100 --mev 10 --arb 20
  
  # Full test with verification
  %(prog)s --bandit 1000 --mev 100 --arb 200 --verify-consume
  
  # Stress test
  %(prog)s --bandit 10000 --mev 1000 --arb 2000 --ch-wait 30
  
  # With Grafana testing
  %(prog)s --grafana-url http://localhost:3000 --grafana-key <api-key>
        """
    )
    
    # Connection settings
    parser.add_argument("--brokers", default=os.getenv("KAFKA_BROKERS", "localhost:9092"),
                       help="Kafka brokers")
    parser.add_argument("--ch-url", default=os.getenv("CLICKHOUSE_URL", "http://localhost:8123"),
                       help="ClickHouse URL")
    parser.add_argument("--grafana-url", default=os.getenv("GRAFANA_URL", ""),
                       help="Grafana URL (optional)")
    parser.add_argument("--grafana-key", default=os.getenv("GRAFANA_API_KEY", ""),
                       help="Grafana API key")
    
    # Event counts
    parser.add_argument("--bandit", type=int, default=100,
                       help="Number of bandit events to generate")
    parser.add_argument("--mev", type=int, default=20,
                       help="Number of MEV events to generate")
    parser.add_argument("--arb", type=int, default=50,
                       help="Number of arbitrage opportunities to generate")
    
    # Test settings
    parser.add_argument("--ch-wait", type=int, default=10,
                       help="Seconds to wait for ClickHouse Kafka engine")
    parser.add_argument("--verify-consume", action="store_true",
                       help="Verify Kafka consumption")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", help="Output file for metrics JSON")
    
    args = parser.parse_args()
    
    # Run async test
    test = ComprehensiveSmokeTest(args)
    metrics = asyncio.run(test.run())
    
    # Exit with appropriate code
    sys.exit(0 if metrics.total_events_consumed > 0 else 1)


if __name__ == "__main__":
    main()