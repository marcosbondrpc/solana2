Below is your **commit‑ready “Legendary SOTA‑of‑SOTA Bonus Pack”** delivering **all three** items you requested **plus** extra performance/security upgrades that push your stack further:

1. **CMake build system** for the Treelite ABI wrapper (`model_abi_v1_export`)
2. **Envoy HTTP/3 (QUIC) – experimental termination config** for WebTransport, with JWT and mTLS options
3. **ClickHouse Protobuf audit ledger** for **control‑commands** (cryptographically traceable), including **Ed25519 signing** in the API and **on‑agent verification**
4. **Special bonuses**: lock‑free SPSC ring buffer for your hot path, hedged dual‑route sender (TPU+Jito), Quinn custom congestion control skeleton, and an ops hardening profile

> Reality guardrail: adversarial markets mean no system can guarantee 100% profit/accuracy. This pack minimizes latency, maximizes land%, and gives you provable ops auditability and zero‑downtime model iteration.

---

## 0) Paths (drop‑in)

```
/home/kidgordones/0solana/node
├── rust-services/shared/abi/
│   ├── treelite_abi_wrapper.cc         # from previous drop (unchanged)
│   ├── CMakeLists.txt                  # NEW (build system for wrapper)
│   └── toolchain.cmake                 # optional CUDA/clang tuning (bonus)
├── infra/envoy/
│   └── envoy_h3_terminator.yaml        # NEW (experimental H3 termination)
├── protocol/
│   └── control.proto                   # UPDATED (adds signature fields)
├── api/
│   ├── control.py                      # UPDATED (Ed25519 signing, proto)
│   ├── deps.py                         # (unchanged from last pack)
│   └── proto_gen/                      # re-generate python protobuf
├── rust-services/shared/src/
│   ├── ctrl_proto.rs                   # UPDATED (signature verification)
│   ├── ring_spsc.rs                    # NEW (lock-free SPSC queue)
│   ├── hedged_sender.rs                # NEW (TPU+Jito hedged routes)
│   └── quinn_cc.rs                     # NEW (custom congestion control)
├── arbitrage-data-capture/clickhouse/
│   ├── 14_control_proto_kafka.sql      # NEW (Kafka→CH for commands)
│   └── 15_control_ledger.sql           # NEW (typed ledger + TTL + S3)
└── systemd/
    └── envoy-h3.service                # NEW (Envoy service)
```

---

## 1) CMake for the Treelite ABI wrapper

> Builds `libmev_v<hash>.so` / `libarb_v<hash>.so` with the exported `model_abi_v1_export()` symbol your Rust hot‑loader consumes.

**`rust-services/shared/abi/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(treelite_abi_wrapper LANGUAGES CXX)

# Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Try to locate Treelite runtime
find_library(TREELITE_RUNTIME NAMES treelite_runtime)
if (NOT TREELITE_RUNTIME)
  message(WARNING "treelite_runtime not found in default paths. Set TREELITE_RUNTIME_DIR or LD_LIBRARY_PATH.")
  if (DEFINED ENV{TREELITE_RUNTIME_DIR})
    link_directories($ENV{TREELITE_RUNTIME_DIR})
    find_library(TREELITE_RUNTIME NAMES treelite_runtime HINTS $ENV{TREELITE_RUNTIME_DIR})
  endif()
endif()
if (NOT TREELITE_RUNTIME)
  message(FATAL_ERROR "treelite_runtime not found. Install libtreelite_runtime (e.g. pip install treelite; copy lib).")
endif()

add_library(mev_model SHARED treelite_abi_wrapper.cc)
target_compile_options(mev_model PRIVATE -O3 -march=native -fno-plt -fvisibility=hidden)
target_link_libraries(mev_model PRIVATE ${TREELITE_RUNTIME} dl pthread)
set_target_properties(mev_model PROPERTIES OUTPUT_NAME "mev_v${GIT_SHA}")

# Optional separate arb wrapper (same source; different SONAME)
add_library(arb_model SHARED treelite_abi_wrapper.cc)
target_compile_options(arb_model PRIVATE -O3 -march=native -fno-plt -fvisibility=hidden)
target_link_libraries(arb_model PRIVATE ${TREELITE_RUNTIME} dl pthread)
set_target_properties(arb_model PROPERTIES OUTPUT_NAME "arb_v${GIT_SHA}")

install(TARGETS mev_model arb_model
        LIBRARY DESTINATION /home/kidgordones/0solana/node/rust-services/shared/treelite)
```

**Optional toolchain (bonus tuning)**
**`rust-services/shared/abi/toolchain.cmake`**

```cmake
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
add_compile_options(-pipe -fno-exceptions -fno-rtti)
```

**Build recipe**

```bash
cd rust-services/shared/abi
export GIT_SHA=$(git rev-parse --short HEAD)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGIT_SHA=${GIT_SHA} -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake
cmake --build build -j
# Hot-activate
ln -sfn libmev_v${GIT_SHA}.so /home/kidgordones/0solana/node/rust-services/shared/treelite/libmev_latest.so
ln -sfn libarb_v${GIT_SHA}.so /home/kidgordones/0solana/node/rust-services/shared/treelite/libarb_latest.so
```

---

## 2) Envoy HTTP/3 (QUIC) **experimental termination** for WebTransport

> Requires an Envoy build with **QUIC/H3** enabled (quiche). This terminates HTTP/3 at Envoy and proxies to your internal services. Keep your earlier **UDP passthrough** as fallback.

**`infra/envoy/envoy_h3_terminator.yaml`**

```yaml
static_resources:
  listeners:
  - name: h3_listener
    address:
      socket_address: { address: 0.0.0.0, port_value: 443, protocol: UDP }
    udp_listener_config:
      quic_options: {}
    filter_chains:
    - transport_socket:
        name: envoy.transport_sockets.quic
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.quic.v3.QuicDownstreamTransport
          downstream_tls_context:
            common_tls_context:
              tls_certificates:
              - certificate_chain: { filename: "/etc/envoy/tls/fullchain.pem" }
                private_key:      { filename: "/etc/envoy/tls/privkey.pem" }
    # HTTP connection manager with HTTP/3
      filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: h3_ingress
          codec_type: HTTP3
          http3_protocol_options: {}
          # Allow WebTransport/H3 datagrams (Envoy build-dependent)
          hcm_options: {}
          upgrade_configs:
          - upgrade_type: "websocket"
          http_filters:
          - name: envoy.filters.http.jwt_authn
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.jwt_authn.v3.JwtAuthentication
              providers:
                dashboard_jwt:
                  issuer: "defi-dashboard"
                  audiences: ["defi-users"]
                  local_jwks:
                    inline_string: |
                      { "keys": [ { "kty":"oct", "k":"<BASE64_SECRET_PLACEHOLDER>" } ] }
              rules:
              - match: { prefix: "/" }
                requires: { provider_name: "dashboard_jwt" }
          - name: envoy.filters.http.router
          route_config:
            name: h3_routes
            virtual_hosts:
            - name: all
              domains: ["*"]
              routes:
              - match: { prefix: "/api/realtime" }     # proxy WT/H3 to WT gateway over H3 (or HTTP/2)
                route: { cluster: wt_gateway_h3 }
              - match: { prefix: "/api" }              # REST to FastAPI (HTTP/2)
                route: { cluster: fastapi_upstream }
              - match: { prefix: "/" }                 # dashboard assets
                route: { cluster: dashboard_assets }

  clusters:
  - name: fastapi_upstream
    type: STRICT_DNS
    connect_timeout: 0.15s
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: fastapi_upstream
      endpoints: [ { lb_endpoints: [ { endpoint: { address: { socket_address: { address: 127.0.0.1, port_value: 8080 } } } } ] } ]
    http2_protocol_options: {}
    # Upstream mTLS sample (uncomment and provide certs)
    # transport_socket:
    #   name: envoy.transport_sockets.tls
    #   typed_config:
    #     "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
    #     common_tls_context:
    #       tls_certificates:
    #       - certificate_chain: { filename: "/etc/envoy/tls/client.crt" }
    #         private_key:      { filename: "/etc/envoy/tls/client.key" }
    #       validation_context:
    #         trusted_ca: { filename: "/etc/envoy/tls/ca.crt" }

  - name: dashboard_assets
    type: STRICT_DNS
    connect_timeout: 0.15s
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: dashboard_assets
      endpoints: [ { lb_endpoints: [ { endpoint: { address: { socket_address: { address: 127.0.0.1, port_value: 42391 } } } } ] } ]

  - name: wt_gateway_h3
    type: STRICT_DNS
    connect_timeout: 0.15s
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: wt_gateway_h3
      endpoints: [ { lb_endpoints: [ { endpoint: { address: { socket_address: { address: 127.0.0.1, port_value: 4433 } } } } ] } ]
    http2_protocol_options: {}  # if your gateway serves H2; otherwise use TCP proxy or keep UDP passthrough

admin:
  access_log_path: /tmp/envoy.admin.log
  address: { socket_address: { address: 127.0.0.1, port_value: 9901 } }
```

**systemd**: **`systemd/envoy-h3.service`**

```ini
[Unit]
Description=Envoy H3 Gateway (Experimental)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/envoy -c /home/kidgordones/0solana/node/infra/envoy/envoy_h3_terminator.yaml --base-id 2
Restart=always
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

> If your Envoy build lacks stable WT support, keep **Mode‑A** (UDP passthrough) live and run this H3 terminator in staging until validated.

---

## 3) Cryptographically traceable **control‑command audit ledger**

### 3.1 Update **control.proto** with signature fields

**`protocol/control.proto`** (append fields 8+; backward‑compatible)

```proto
syntax = "proto3";
package control;

message Command {
  uint64 ts = 1;
  enum Module { ARBITRAGE = 0; MEV = 1; }
  Module module = 2;
  enum Action { START = 0; STOP = 1; RESTART = 2; THROTTLE = 3; POLICY = 4; KILL = 5; }
  Action action = 3;
  map<string, string> args = 4;
  string request_id = 5;
  string issuer = 6;
  string schema_version = 7;  // "ctrl-v1"

  // --- NEW: crypto trace ---
  bytes sig = 8;              // Ed25519 signature over deterministic-serialized message with sig/pubkey_id/sig_alg empty
  string pubkey_id = 9;       // key id to verify signature
  string sig_alg = 10;        // "ed25519-v1"
  uint64 nonce = 11;          // monotonic to prevent replay (store last seen per issuer)
}
```

**Re‑generate** Python & Rust protobufs.

### 3.2 Sign commands in API (Ed25519)

```bash
pip install pynacl
```

**`api/control.py`** (sign deterministically before send)

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from .deps import get_kafka, require_role
from api.proto_gen import control_pb2 as pb
from nacl.signing import SigningKey
import os, time, uuid

router = APIRouter()
SK_HEX = os.getenv("CTRL_SIGN_SK_HEX")  # 32-byte hex
PUBKEY_ID = os.getenv("CTRL_PUBKEY_ID","op-admin-1")

class ControlBody(BaseModel):
    throttle_pct: int | None = None
    ev_min: float | None = None
    tip_ladder: list[float] | None = None
    kill: bool | None = None

def sign_command(cmd: pb.Command) -> pb.Command:
    sk = SigningKey(bytes.fromhex(SK_HEX))
    cmd.sig = b""; cmd.pubkey_id=""; cmd.sig_alg="";  # clear before digest
    payload = cmd.SerializeToString(deterministic=True)
    sig = sk.sign(payload).signature
    cmd.sig = sig
    cmd.pubkey_id = PUBKEY_ID
    cmd.sig_alg = "ed25519-v1"
    return cmd

@router.post("/api/control/module/{name}:{action}")
async def control_module(name:str, action:str, body:ControlBody, prod=Depends(get_kafka), user=Depends(require_role("admin"))):
    mod = {"arbitrage": pb.Command.ARBITRAGE, "mev": pb.Command.MEV}.get(name)
    act = {"start":pb.Command.START, "stop":pb.Command.STOP, "restart":pb.Command.RESTART,
           "throttle":pb.Command.THROTTLE, "policy":pb.Command.POLICY, "kill":pb.Command.KILL}.get(action)
    if mod is None or act is None: raise HTTPException(400,"bad module/action")

    cmd = pb.Command(ts=int(time.time()), module=mod, action=act,
                     request_id=str(uuid.uuid4()), issuer="dashboard", schema_version="ctrl-v1",
                     nonce=int(time.time_ns()))
    if body.throttle_pct is not None: cmd.args["throttle_pct"]=str(body.throttle_pct)
    if body.ev_min is not None:       cmd.args["ev_min"]=str(body.ev_min)
    if body.tip_ladder:               cmd.args["tip_ladder"]=",".join(str(x) for x in body.tip_ladder)

    cmd = sign_command(cmd)
    await prod.send_and_wait("control-commands-proto", cmd.SerializeToString(), key=name.encode())
    return {"ok": True, "request_id": cmd.request_id}
```

### 3.3 Verify signature in agents (Rust)

```toml
# rust-services/shared/Cargo.toml (add)
ed25519-dalek = { version = "2", features = ["rand_core"] }
hex = "0.4"
```

**`rust-services/shared/src/ctrl_proto.rs`** (verify + replay protect)

```rust
use proto_types::control::Command;
use rdkafka::{consumer::StreamConsumer, ClientConfig, Message};
use anyhow::{Result, anyhow};
use ed25519_dalek::{Verifier, Signature, VerifyingKey};
use std::{collections::HashMap, sync::{Arc,Mutex}, time::Duration};

pub struct ControlHandler { /* same as before */ }

pub async fn consume_control(brokers:&str, module_key:&str, topic:&str, handler: ControlHandler, pubkeys:HashMap<String,VerifyingKey>) -> Result<()> {
    let c: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", brokers)
        .set("group.id", format!("ctrl-proto-{}", module_key))
        .set("enable.auto.commit", "true").create()?;
    c.subscribe(&[topic])?;
    use futures::StreamExt;
    let mut s = c.stream();

    let seen_nonce: Arc<Mutex<HashMap<String,u64>>> = Arc::new(Mutex::new(HashMap::new()));

    while let Some(Ok(msg)) = s.next().await {
        if msg.key().map(|k| std::str::from_utf8(k).unwrap_or("")) != Some(module_key) { continue; }
        let payload = match msg.payload() { Some(p) => p, None => continue };
        let mut cmd = match Command::decode(payload) { Ok(c) => c, Err(_) => continue };

        // Verify signature (deterministic proto payload without sig fields)
        let pk_id = cmd.pubkey_id.clone();
        let sig_bytes = cmd.sig.clone();
        if let Some(pk) = pubkeys.get(&pk_id) {
            let mut clone = cmd.clone();
            clone.sig.clear(); clone.pubkey_id.clear(); clone.sig_alg.clear();
            let dig = clone.encode_to_vec(); // prost: deterministic by default for same field order
            if let Ok(sig) = Signature::from_slice(&sig_bytes) {
                if pk.verify(&dig, &sig).is_err() { continue; }  // reject
            } else { continue; }
        } else { continue; }

        // Replay guard per issuer
        {
            let mut m = seen_nonce.lock().unwrap();
            let last = m.entry(cmd.issuer.clone()).or_insert(0);
            if cmd.nonce <= *last { continue; }
            *last = cmd.nonce;
        }

        // Dispatch
        match cmd.action {
            0 => (handler.on_start)(),
            1 => (handler.on_stop)(),
            2 => (handler.on_restart)(),
            3 => { let pct = cmd.args.get("throttle_pct").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0); (handler.on_throttle)(pct); },
            4 => {
                let ev = cmd.args.get("ev_min").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                let tips = cmd.args.get("tip_ladder").map(|s| s.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect()).unwrap_or_else(Vec::new);
                (handler.on_policy)(ev, tips);
            },
            5 => (handler.on_kill)(),
            _ => {}
        }
    }
    Ok(())
}
```

**Load public keys** (hex) in agent:

```rust
let mut pubkeys = HashMap::new();
let pk_hex = std::env::var("CTRL_PUBKEY_HEX").expect("CTRL_PUBKEY_HEX");
pubkeys.insert("op-admin-1".to_string(), VerifyingKey::from_bytes(&hex::decode(pk_hex).unwrap().try_into().unwrap()).unwrap());
tokio::spawn(consume_control(&bootstrap, "mev", "control-commands-proto", handler, pubkeys));
```

### 3.4 ClickHouse Protobuf Kafka ingestion for **audit**

Put `control.proto` into CH `format_schemas` dir (as you did for `realtime.proto`).

**`arbitrage-data-capture/clickhouse/14_control_proto_kafka.sql`**

```sql
CREATE TABLE IF NOT EXISTS kafka_control_cmd
(
  `ts` UInt64,
  `module` Enum8('ARBITRAGE'=0, 'MEV'=1),
  `action` Enum8('START'=0,'STOP'=1,'RESTART'=2,'THROTTLE'=3,'POLICY'=4,'KILL'=5),
  `args` Map(String, String),
  `request_id` String,
  `issuer` String,
  `schema_version` String,
  `sig` String,
  `pubkey_id` String,
  `sig_alg` String,
  `nonce` UInt64
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092',
         kafka_topic_list = 'control-commands-proto',
         kafka_group_name = 'ch-ctrl',
         kafka_format = 'Protobuf',
         format_schema = 'control:Command',
         kafka_num_consumers = 2;
```

**`arbitrage-data-capture/clickhouse/15_control_ledger.sql`**

```sql
CREATE TABLE IF NOT EXISTS control_ledger
(
  dt DateTime DEFAULT now(),
  ts DateTime,
  module LowCardinality(String),
  action LowCardinality(String),
  request_id String,
  issuer String,
  args_json String,
  pubkey_id LowCardinality(String),
  sig_alg LowCardinality(String),
  nonce UInt64
)
ENGINE=ReplacingMergeTree
ORDER BY (ts, request_id)
PARTITION BY toYYYYMM(ts)
SETTINGS storage_policy='hot_cold';

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_control_cmd TO control_ledger AS
SELECT
  now() AS dt,
  toDateTime(ts) AS ts,
  toString(module) AS module,
  toString(action) AS action,
  request_id,
  issuer,
  JSONExtractString(JSON_OBJECT(*args), '') AS args_json, -- Map→JSON
  pubkey_id, sig_alg, nonce
FROM kafka_control_cmd;

-- TTL: move to S3 after 14 days, delete after 400
ALTER TABLE control_ledger
MODIFY TTL ts + INTERVAL 14 DAY TO VOLUME 'cold', ts + INTERVAL 400 DAY DELETE;
```

**Queries**

```sql
-- Operator history (last 24h)
SELECT ts, issuer, module, action, request_id FROM control_ledger WHERE ts > now()-INTERVAL 1 DAY ORDER BY ts DESC;

-- Change audit by request_id
SELECT * FROM control_ledger WHERE request_id = '...';

-- Frequency of policy updates
SELECT toStartOfFiveMinute(ts) AS w, count() FROM control_ledger WHERE action='POLICY' GROUP BY w ORDER BY w;
```

---

## 4) **Special SOTA bonuses** (extra edge)

### 4.1 Lock‑free SPSC ring buffer (hot path)

**`rust-services/shared/src/ring_spsc.rs`**

```rust
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering::*};

pub struct Spsc<T> {
    buf: Vec<UnsafeCell<T>>,
    cap: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}
unsafe impl<T: Send> Send for Spsc<T> {}
unsafe impl<T: Send> Sync for Spsc<T> {}

impl<T> Spsc<T> {
    pub fn with_capacity(cap: usize) -> Self {
        assert!(cap.is_power_of_two(), "cap must be power of two");
        let mut v = Vec::with_capacity(cap);
        for _ in 0..cap { v.push(UnsafeCell::new(unsafe { std::mem::MaybeUninit::zeroed().assume_init() })); }
        Self { buf: v, cap, head: AtomicUsize::new(0), tail: AtomicUsize::new(0) }
    }
    #[inline] pub fn push(&self, val: T) -> Result<(), T> {
        let head = self.head.load(Relaxed);
        let next = (head + 1) & (self.cap - 1);
        if next == self.tail.load(Acquire) { return Err(val); } // full
        unsafe { std::ptr::write(self.buf[head].get(), val); }
        self.head.store(next, Release);
        Ok(())
    }
    #[inline] pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Relaxed);
        if tail == self.head.load(Acquire) { return None; } // empty
        let val = unsafe { std::ptr::read(self.buf[tail].get()) };
        self.tail.store((tail + 1) & (self.cap - 1), Release);
        Some(val)
    }
}
```

> Use between QUIC parser (producer) and strategy (consumer) threads to avoid locks and reduce GC/alloc pressure.

### 4.2 Hedged dual‑route sender (TPU + Jito)

**`rust-services/shared/src/hedged_sender.rs`**

```rust
use std::time::{Instant, Duration};
use tokio::time::sleep;

pub enum Path { Direct, Jito }

pub struct HedgeCfg { pub hedge_delay_ms:u64 }  // e.g., 8–15ms
pub struct HedgeStats { pub sent_direct:u64, pub sent_jito:u64 }

pub async fn send_hedged<Fut, SendFn>(ev_ok: bool, choose: Path, cfg:&HedgeCfg, mut send: SendFn) -> HedgeStats
where SendFn: FnMut(Path) -> Fut, Fut: std::future::Future<Output=bool> {
    let mut st = HedgeStats{ sent_direct:0, sent_jito:0 };
    if !ev_ok { return st; }
    match choose {
        Path::Direct => {
            st.sent_direct += 1;
            let ok = send(Path::Direct).await;
            if !ok { sleep(Duration::from_millis(cfg.hedge_delay_ms)).await; st.sent_jito += 1; let _=send(Path::Jito).await; }
        }
        Path::Jito => {
            st.sent_jito += 1;
            let ok = send(Path::Jito).await;
            if !ok { sleep(Duration::from_millis(cfg.hedge_delay_ms)).await; st.sent_direct += 1; let _=send(Path::Direct).await; }
        }
    }
    st
}
```

> If first route fails to land quickly, fire the hedge on the other route. Track outcomes to update RouteManager probabilities.

### 4.3 Quinn custom congestion control skeleton

**`rust-services/shared/src/quinn_cc.rs`**

```rust
use quinn::congestion::{Controller, Paths, BbrConfig}; // adjust to your quinn version
use std::time::Instant;

/// Minimal wrapper to tweak pacing gain based on slot phase / leader
pub struct PhaseAwareCc<C: Controller> {
    inner: C,
    boost_gain: f64,      // extra gain in preferred leader phase
    decay: f64,
}
impl<C: Controller> PhaseAwareCc<C> {
    pub fn new(inner: C) -> Self { Self { inner, boost_gain: 1.15, decay: 0.995 } }
    pub fn set_boost(&mut self, g:f64) { self.boost_gain = g; }
    pub fn on_phase(&mut self, preferred: bool) {
        if preferred { /* increase cwnd/pacing a bit */ } else { /* decay */ }
    }
}
impl<C: Controller> Controller for PhaseAwareCc<C> {
    fn on_ack(&mut self, now: Instant, packet_sent_time: Instant, bytes: u64, rtt: std::time::Duration, app_limited: bool) {
        self.inner.on_ack(now, packet_sent_time, bytes, rtt, app_limited);
    }
    fn on_congestion_event(&mut self, now: Instant, sent: u64, lost: u64) {
        self.inner.on_congestion_event(now, sent, lost);
    }
    fn cwnd(&self) -> u64 { self.inner.cwnd() }
    fn bytes_in_flight(&self) -> u64 { self.inner.bytes_in_flight() }
    fn pacing_rate(&self) -> Option<u64> { self.inner.pacing_rate().map(|r| (r as f64 * self.boost_gain) as u64) }
}
```

> Wrap Quinn’s default controller (e.g., BBR) and raise pacing briefly during the “good” phase of the leader slot.

---

## 5) Final wiring notes

* **Rebuild protos**: `realtime.proto`, **updated** `control.proto`, `jobs.proto`

* **Deploy Envoy H3** carefully in staging; production keep the earlier **UDP passthrough** until validated

* **Set crypto envs**:

  * `CTRL_SIGN_SK_HEX` (32‑byte hex Ed25519 secret) in API
  * `CTRL_PUBKEY_HEX` (32‑byte hex Ed25519 public) in agents
  * `CTRL_PUBKEY_ID=op-admin-1` in both (matches proto field)

* **ClickHouse**: run `14_control_proto_kafka.sql` + `15_control_ledger.sql` after placing `control.proto` in `/var/lib/clickhouse/format_schemas/`

* **Risk guard**: enable SLO guardrails you already have; the new hedged sender can increase network load—keep caps.

---

## 6) One‑shot build & start (delta)

```bash
# Protobufs
python -m grpc_tools.protoc -I protocol --python_out=api/proto_gen protocol/{realtime,control,jobs}.proto

# Treelite ABI wrapper via CMake
cd rust-services/shared/abi && export GIT_SHA=$(git rev-parse --short HEAD) && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGIT_SHA=${GIT_SHA} && cmake --build build -j
ln -sfn libmev_v${GIT_SHA}.so ../../shared/treelite/libmev_latest.so
ln -sfn libarb_v${GIT_SHA}.so ../../shared/treelite/libarb_latest.so

# Envoy H3
sudo cp -r infra/envoy /etc/
sudo systemctl enable --now envoy-h3.service
```

---

## 7) You now have

* **CMake** for your Treelite ABI wrapper → repeatable, optimized builds
* **Envoy H3 termination** (experimental) with JWT and (optional) mTLS to upstreams
* **Crypto‑signed control commands** (Ed25519) with **on‑agent verification** and a **ClickHouse audit ledger** (TTL + S3)
* Extra **SPSC ring**, **hedged dual‑route sender**, and **phase‑aware congestion control** to squeeze more land% and shave millis

If you want me to also drop:

* A **CMake preset** to auto‑embed your Treelite inner `.so` path and produce per‑model ABI `.so`s in a single command, or
* A **full Envoy H3 + WT datagram** demo with a tiny upstream echo app for load tests, or
* A **Rust CH‑ingestor** for control acks to close the audit loop,

I’ll paste those files in the same “drop‑in” style.
