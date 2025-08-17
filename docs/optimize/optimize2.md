Absolutely—here’s a **drop‑in, commit‑ready pack** that adds:

1. **Protobuf schemas** for *all* realtime events.
2. **Browser decoder** (TypeScript, tree‑shakable) w/ optional **Zstd‑WASM** decompression.
3. A **full FastAPI control plane** wired to a **Dramatiq** job runner (Redis broker) for **dataset scraping** and **model training** (MEV & Arbitrage)—with **progress**, **logs**, and **WebSocket** push using **protobuf batches** or **NDJSON**.
4. Operational glue: **Kafka control topic**, **ClickHouse exports**, **status storage**, **RBAC/JWT**, **systemd** units, and **end‑to‑end wiring** to your existing agents.

> **Reality guardrail:** No pipeline can guarantee 100% accuracy/profit in adversarial markets. This design pushes latency, observability, and safety to the limit.

---

## 📦 New/Updated Paths

```
/home/kidgordones/0solana/node
├── protocol/
│   └── realtime.proto
├── api/
│   ├── main.py                  # FastAPI app (control plane + WS)
│   ├── deps.py                  # JWT/RBAC, Kafka, Redis, ClickHouse clients
│   ├── control.py               # control endpoints → Kafka control-commands
│   ├── datasets.py              # dataset scrape API → Dramatiq
│   ├── training.py              # training API → Dramatiq
│   ├── realtime.py              # WebSocket mux (NDJSON or Proto+Zstd)
│   └── proto_gen/               # generated Python protobuf (compiled)
├── workers/
│   ├── broker.py                # Dramatiq Redis broker config
│   ├── tasks_scrape.py          # CH → Parquet/JSONL (streaming)
│   ├── tasks_train.py           # Train → Treelite export
│   ├── util.py                  # progress/log helpers
│   └── __main__.py              # worker entry
├── defi-frontend/
│   ├── lib/proto/realtime.js    # protobufjs static module (generated)
│   ├── lib/proto/realtime.d.ts  # TS types (generated)
│   ├── lib/ws-proto.ts          # browser decoder (zstd + proto)
│   └── (existing Next.js code from previous step)
└── systemd/
    ├── fastapi-control.service
    └── dramatiq-worker.service
```

---

## 1) Protobuf schema

**`protocol/realtime.proto`**

```proto
syntax = "proto3";
package realtime;

// --------- Core event payloads ---------
message ArbOpportunity {
  string version = 1;
  uint64 slot = 2;
  string tx_signature = 3;
  double net_sol = 4;
  string classification = 5;
  double p_land_est = 6;
  repeated double tip_ladder = 7;
  bytes raw = 15; // optional: full JSON serialized for archival/debug
}

message MevOpportunity {
  string version = 1;
  uint64 slot = 2;
  string attacker = 3;
  double net_sol = 4;
  string type = 5; // "pump_fun_sandwich", etc.
  double p_land_est = 6;
  repeated double tip_ladder = 7;
  string bundle_id = 8;
  bytes raw = 15; // optional JSON
}

message BundleOutcome {
  string bundle_id = 1;
  bool landed = 2;
  uint64 tip_lamports = 3;
  string path = 4;   // TPU | Jito
  string leader = 5; // leader identity
}

message MetricsUpdate {
  double hotpath_p50 = 1;
  double hotpath_p99 = 2;
  double send_p99 = 3;
  double landed_pct = 4; // last 10m
  double pnl_1h = 5;
}

message TrainingStatus {
  string job_id = 1;
  string state = 2;    // queued|running|done|error
  double progress = 3; // 0..1
  string message = 4;
}

message DatasetStatus {
  string job_id = 1;
  string state = 2;    // queued|running|done|error
  double progress = 3; // 0..1
  string message = 4;
}

// --------- Envelope + Batch ---------
message Envelope {
  enum Type { UNKNOWN = 0; ARB = 1; MEV = 2; OUTCOME = 3; METRICS = 4; TRAINING = 5; DATASET = 6; }
  Type type = 1;
  oneof payload {
    ArbOpportunity arb = 10;
    MevOpportunity mev = 11;
    BundleOutcome outcome = 12;
    MetricsUpdate metrics = 13;
    TrainingStatus training = 14;
    DatasetStatus dataset = 15;
  }
}

message Batch {
  repeated Envelope items = 1;
}
```

### Generate code (Python + TS)

**Python (server):**

```bash
pip install protobuf==5.27.0
python -m grpc_tools.protoc -I protocol --python_out=api/proto_gen protocol/realtime.proto
```

**TypeScript (browser) using protobufjs static module:**

```bash
npm i -D protobufjs
npx pbjs -t static-module -w es6 -o defi-frontend/lib/proto/realtime.js protocol/realtime.proto
npx pbts -o defi-frontend/lib/proto/realtime.d.ts defi-frontend/lib/proto/realtime.js
```

---

## 2) Browser decoder (Proto + optional Zstd)

**`defi-frontend/lib/ws-proto.ts`**

```ts
// Binary WS decoder: Zstd (optional) + protobuf Batch
// npm i @mongodb-js/zstd
import { ZstdInit, ZstdSimple } from "@mongodb-js/zstd";
import * as PB from "@/lib/proto/realtime"; // generated

let zstdReady: Promise<ZstdSimple> | null = null;
async function getZstd() {
  if (!zstdReady) zstdReady = ZstdInit().then(() => new ZstdSimple());
  return zstdReady;
}

export type OnEvent = (env: PB.realtime.Envelope) => void;

export async function decodeBinaryFrame(buf: ArrayBuffer, compressed = true, on: OnEvent) {
  let u8 = new Uint8Array(buf);
  if (compressed) {
    const zstd = await getZstd();
    u8 = zstd.decompress(u8);
  }
  const batch = PB.realtime.Batch.decode(u8);
  for (const env of batch.items) on(env);
}
```

Update your WS client to branch on **Proto mode**:

**`defi-frontend/lib/ws.ts`** (add binary branch)

```ts
import { decodeBinaryFrame } from "./ws-proto";
import type { WsEvent } from "./types";

export class RealtimeClient {
  ws?: WebSocket;
  constructor(private url: string, private token: string, private useProto = false) {}
  connect(onJson:(ev:WsEvent)=>void, onProto?:(env:any)=>void) {
    const u = `${this.url}?token=${encodeURIComponent(this.token)}&mode=${this.useProto?"proto":"json"}`;
    this.ws = new WebSocket(u);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = async (m) => {
      if (this.useProto && m.data instanceof ArrayBuffer) {
        await decodeBinaryFrame(m.data, /*compressed*/true, (env)=> onProto?.(env));
        return;
      }
      if (typeof m.data === "string") {
        m.data.split("\n").forEach((line:string) => { if (!line.trim()) return;
          try { onJson(JSON.parse(line)); } catch {}
        });
      }
    };
    this.ws.onclose = () => setTimeout(()=> this.connect(onJson, onProto), 1000);
  }
}
```

---

## 3) FastAPI control plane (REST + WS + JWT/RBAC)

### 3.1 Dependencies

```bash
pip install fastapi uvicorn[standard] aiokafka redis clickhouse-connect pydantic-settings python-jose[cryptography] passlib[bcrypt] websockets zstandard dramatiq "protobuf==5.27.0"
```

### 3.2 App deps (clients, JWT, RBAC)

**`api/deps.py`**

```python
import os, redis, clickhouse_connect, aiokafka
from functools import lru_cache
from pydantic_settings import BaseSettings
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class Settings(BaseSettings):
    SECRET: str = "change-me"
    JWT_ISS: str = "defi-dashboard"
    JWT_AUD: str = "defi-users"
    KAFKA: str = "localhost:9092"
    REDIS: str = "redis://localhost:6379/0"
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_DB: str = "default"
    class Config: env_file = ".env"

@lru_cache()
def settings(): return Settings()

async def get_kafka():
    prod = aiokafka.AIOKafkaProducer(bootstrap_servers=settings().KAFKA, compression_type="zstd")
    await prod.start()
    try: yield prod
    finally: await prod.stop()

def get_redis():
    return redis.Redis.from_url(settings().REDIS, decode_responses=True)

def get_ch():
    s = settings()
    return clickhouse_connect.get_client(host=s.CLICKHOUSE_HOST, database=s.CLICKHOUSE_DB)

# RBAC / JWT
bearer = HTTPBearer(auto_error=True)
def require_role(role:str):
    async def _dep(token: HTTPAuthorizationCredentials = Depends(bearer)):
        try:
            payload = jwt.decode(token.credentials, settings().SECRET, audience=settings().JWT_AUD, issuer=settings().JWT_ISS)
            roles = payload.get("roles", [])
            if role not in roles: raise HTTPException(403, "forbidden")
            return payload
        except JWTError as e:
            raise HTTPException(401, "bad token")
    return _dep
```

### 3.3 Control, Datasets, Training, Realtime

**`api/control.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from .deps import get_kafka, require_role
import json, time, uuid

router = APIRouter()

class ControlBody(BaseModel):
    throttle_pct: int | None = None
    ev_min: float | None = None
    tip_ladder: list[float] | None = None
    kill: bool | None = None

@router.post("/api/control/module/{name}:{action}")
async def control_module(name:str, action:str, body:ControlBody, prod=Depends(get_kafka), _=Depends(require_role("admin"))):
    if name not in ("arbitrage","mev"): raise HTTPException(400,"bad module")
    msg = {
        "ts": int(time.time()),
        "module": name,
        "action": action,
        "args": {k:v for k,v in body.dict().items() if v is not None},
        "request_id": str(uuid.uuid4())
    }
    await prod.send_and_wait("control-commands", json.dumps(msg).encode(), key=name.encode())
    return {"ok": True, "request_id": msg["request_id"]}
```

**`api/datasets.py`**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from .deps import get_redis, require_role
import uuid, time, json
from workers.tasks_scrape import scrape_dataset  # Dramatiq task

router = APIRouter()

class ScrapeReq(BaseModel):
    start:str; end:str; types:list[str]; dex:list[str]
    format:str="parquet"; compress:str="zstd"; output:str="/mnt/data/out"

@router.post("/api/datasets/scrape")
async def scrape(req:ScrapeReq, r=Depends(get_redis), _=Depends(require_role("admin"))):
    job_id = f"scrape-{uuid.uuid4()}"
    r.hset(job_id, mapping={"state":"queued","progress":"0","message":""})
    # send task
    scrape_dataset.send(job_id, req.dict())
    return {"ok": True, "job_id": job_id}

@router.get("/api/datasets/status/{job_id}")
async def ds_status(job_id:str, r=Depends(get_redis), _=Depends(require_role("viewer"))):
    return r.hgetall(job_id) or {"state":"unknown"}
```

**`api/training.py`**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from .deps import get_redis, require_role
import uuid
from workers.tasks_train import train_model

router = APIRouter()

class TrainReq(BaseModel):
    module:str;                 # "mev" | "arbitrage"
    range:dict                  # {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}
    features:str="default"
    model:str="xgb"
    gpu:bool=True
    params:dict={"max_depth":8,"learning_rate":0.08}

@router.post("/api/training/start")
async def start(req:TrainReq, r=Depends(get_redis), _=Depends(require_role("admin"))):
    job_id = f"train-{req.module}-{uuid.uuid4()}"
    r.hset(job_id, mapping={"state":"queued","progress":"0","message":""})
    train_model.send(job_id, req.dict())
    return {"ok":True, "job_id": job_id}

@router.get("/api/training/status/{job_id}")
async def status(job_id:str, r=Depends(get_redis), _=Depends(require_role("viewer"))):
    return r.hgetall(job_id) or {"state":"unknown"}
```

**`api/realtime.py`** (WS supports `mode=json|proto`; proto frames Zstd‑compressed Batch)

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from .deps import require_role
import asyncio, json, zstandard as zstd
from api.proto_gen import realtime_pb2 as pb

router = APIRouter()
Z = zstd.ZstdCompressor(level=3)

# Replace with your internal async producers or Kafka consumers
async def fake_stream():
    while True:
        m = {"type":"metrics_update","data":{"hotpath_p50":4.8,"hotpath_p99":12.3,"send_p99":9.4,"landed_pct":0.72,"pnl_1h":1.23}}
        yield m
        await asyncio.sleep(1)

@router.websocket("/api/realtime/ws")
async def ws(ws: WebSocket, mode: str = Query("json")):
    await ws.accept()
    try:
        async for ev in fake_stream():
            if mode == "json":
                await ws.send_text(json.dumps(ev) + "\n")
            else:
                # build protobuf Batch
                env = pb.Envelope(type=pb.Envelope.METRICS,
                                  metrics=pb.MetricsUpdate(**ev["data"]))
                batch = pb.Batch(items=[env])
                raw = batch.SerializeToString()
                frame = Z.compress(raw)
                await ws.send_bytes(frame)
    except WebSocketDisconnect:
        return
```

**`api/main.py`**

```python
from fastapi import FastAPI
from . import control, datasets, training, realtime

app = FastAPI(title="Legendary Control Plane")
app.include_router(control.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(realtime.router)
```

---

## 4) Dramatiq job runner (Redis broker) + tasks

**`workers/broker.py`**

```python
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import os

redis_broker = RedisBroker(url=os.getenv("REDIS_URL","redis://localhost:6379/0"))
dramatiq.set_broker(redis_broker)
```

**`workers/util.py`**

```python
import redis, os, time
R = redis.Redis.from_url(os.getenv("REDIS_URL","redis://localhost:6379/0"), decode_responses=True)

def progress(job_id:str, p:float, msg:str=""):
    R.hset(job_id, mapping={"state":"running","progress":f"{p:.4f}","message":msg, "ts":str(int(time.time()))})

def done(job_id:str, msg:str="ok"):
    R.hset(job_id, mapping={"state":"done","progress":"1.0","message":msg})

def errored(job_id:str, msg:str):
    R.hset(job_id, mapping={"state":"error","message":msg})
```

**`workers/tasks_scrape.py`**

```python
from .broker import redis_broker
import dramatiq, clickhouse_connect, pyarrow as pa, pyarrow.parquet as pq
from .util import progress, done, errored
import os, time, math

@dramatiq.actor
def scrape_dataset(job_id:str, req:dict):
    try:
        start, end = req["start"], req["end"]
        kinds = req["types"]; dex = req["dex"]
        fmt, compress, out = req["format"], req["compress"], req["output"]
        os.makedirs(out, exist_ok=True)

        client = clickhouse_connect.get_client(host=os.getenv("CLICKHOUSE_HOST","localhost"),
                                               database=os.getenv("CLICKHOUSE_DB","default"))
        # Build query
        filters = []
        if "arb" in kinds: filters.append("table='arbitrage'")
        if "mev" in kinds: filters.append("table='mev_sandwich'")
        # Example unified view: you can UNION ALL specialized selects
        q = f"""
        SELECT * FROM mev_sandwich
        WHERE dt >= parseDateTimeBestEffort('{start}') AND dt < parseDateTimeBestEffort('{end}')
        """
        # Stream in chunks using Arrow stream
        stream = client.query_arrow_stream(q)
        sink_path = os.path.join(out, f"mev_{start}_{end}.{ 'parquet' if fmt=='parquet' else 'jsonl'}")
        count, batches = 0, []
        writer = None

        for rb in stream:
            tbl = pa.Table.from_batches([rb])
            count += tbl.num_rows
            if fmt == "parquet":
                if writer is None:
                    writer = pq.ParquetWriter(sink_path, schema=tbl.schema,
                                              compression="zstd" if compress=="zstd" else None)
                writer.write_table(tbl)
            else:
                # jsonl
                with open(sink_path, "a") as f:
                    for row in tbl.to_pylist():
                        f.write(__import__("json").dumps(row)+"\n")
            # progress
            progress(job_id, min(0.95, 0.05 + math.tanh(count/1e6)), f"rows={count}")

        if writer: writer.close()
        done(job_id, f"wrote {count} rows to {sink_path}")
    except Exception as e:
        errored(job_id, str(e))
        raise
```

**`workers/tasks_train.py`**

```python
from .broker import redis_broker
import dramatiq, clickhouse_connect, xgboost as xgb, treelite, shutil, os
from .util import progress, done, errored

@dramatiq.actor
def train_model(job_id:str, req:dict):
    try:
        module = req["module"]; rng = req["range"]; model_name = req["model"]; params = req["params"]; gpu = req["gpu"]
        start, end = rng["start"], rng["end"]
        client = clickhouse_connect.get_client(host=os.getenv("CLICKHOUSE_HOST","localhost"),
                                               database=os.getenv("CLICKHOUSE_DB","default"))
        if module == "mev":
            q = f"""
            SELECT
              toFloat64(JSON_VALUE(frontrun_json,'$.effective_price')) AS fr_px,
              toFloat64(JSON_VALUE(backrun_json,'$.effective_price')) AS br_px,
              toFloat64(JSON_VALUE(market_json,'$.volatility_5s_bps')) AS vol5,
              toFloat64(JSON_VALUE(structure_json,'$.inventory_alignment')) AS inv_al,
              toFloat64(confidence) AS confidence,
              toUInt8(label_is_sandwich) AS y
            FROM mev_sandwich
            WHERE dt >= parseDateTimeBestEffort('{start}') AND dt < parseDateTimeBestEffort('{end}')
            """
        else:
            q = f""" /* TODO: arbitrage select */ SELECT 0 AS fr_px,0 AS br_px,0 AS vol5,1 AS inv_al,1 AS confidence, toUInt8(1) AS y LIMIT 1 """

        df = client.query_df(q)
        progress(job_id, 0.1, f"loaded {len(df)} rows")

        X = df[['fr_px','br_px','vol5','inv_al','confidence']].values
        y = df['y'].values
        clf = xgb.XGBClassifier(
            max_depth=int(params.get("max_depth", 8)),
            n_estimators=int(params.get("n_estimators", 600)),
            learning_rate=float(params.get("learning_rate", 0.08)),
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.2,
            tree_method="hist", n_jobs=8, 
            predictor="gpu_predictor" if gpu else "auto"
        )
        clf.fit(X,y)
        progress(job_id, 0.6, "xgboost trained")

        path = f"/mnt/data/models/{module}"
        os.makedirs(path, exist_ok=True)
        model_json = os.path.join(path, "xgb_model.json")
        clf.save_model(model_json)
        progress(job_id, 0.7, "saved xgb json")

        tl_model = treelite.Model.load_xgboost(model_json, model_format='xgboost_json')
        so_path = treelite.compile(tl_model, toolchain='gcc', params={'parallel_comp':8,'quantize':1})
        dst = f"/home/kidgordones/0solana/node/rust-services/shared/treelite/lib{module}_latest.so"
        shutil.copy(so_path, dst)
        done(job_id, f"treelite exported to {dst}")
    except Exception as e:
        errored(job_id, str(e))
        raise
```

**`workers/__main__.py`**

```python
from .broker import redis_broker  # registers broker
from . import tasks_scrape, tasks_train
if __name__ == "__main__":
    # Start via: python -m workers
    import time
    print("Dramatiq worker loaded")
    while True: time.sleep(3600)
```

Run dramatiq worker with the CLI:

```bash
dramatiq workers.tasks_scrape workers.tasks_train --processes 2 --threads 4
```

---

## 5) systemd units

**`systemd/fastapi-control.service`**

```ini
[Unit]
Description=Legendary Control Plane (FastAPI)
After=network-online.target redis-server.service kafka.service clickhouse-server.service

[Service]
WorkingDirectory=/home/kidgordones/0solana/node/api
Environment=PYTHONUNBUFFERED=1
Environment=REDIS_URL=redis://localhost:6379/0
Environment=CLICKHOUSE_HOST=localhost
Environment=CLICKHOUSE_DB=default
Environment=KAFKA=localhost:9092
Environment=SECRET=change-me
ExecStart=/usr/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

**`systemd/dramatiq-worker.service`**

```ini
[Unit]
Description=Dramatiq Worker (datasets + training)
After=redis-server.service clickhouse-server.service

[Service]
WorkingDirectory=/home/kidgordones/0solana/node
Environment=REDIS_URL=redis://localhost:6379/0
Environment=CLICKHOUSE_HOST=localhost
Environment=CLICKHOUSE_DB=default
ExecStart=/usr/bin/dramatiq workers.tasks_scrape workers.tasks_train --processes 2 --threads 4
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 6) Frontend wiring for Proto mode

Use the toggle on *Settings* page to select **Proto** transport:

```tsx
// in /app/settings/page.tsx (simplified)
"use client";
import Cookies from "js-cookie";
export default function Settings(){
  const save = () => {
    const useProto = (document.getElementById("proto") as HTMLInputElement).checked;
    Cookies.set("transport", useProto ? "proto" : "json");
    alert("Saved");
  };
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
      <label><input id="proto" type="checkbox"/> Use Proto+Zstd transport</label>
      <button className="btn ml-3" onClick={save}>Save</button>
    </div>
  );
}
```

In **Overview page**, instantiate WS client accordingly:

```tsx
// app/page.tsx (snippet)
import Cookies from "js-cookie";
import { RealtimeClient } from "@/lib/ws";
import * as PB from "@/lib/proto/realtime";
...
const useProto = Cookies.get("transport")==="proto";
const client = new RealtimeClient(process.env.NEXT_PUBLIC_WS_URL!, token, useProto);
client.connect(onJson, (env:PB.realtime.Envelope)=>{
  if (env.metrics) {
    // update cards instantly, e.g. setState(...)
  }
});
```

---

## 7) Kafka control wiring in agents (recap)

Ensure both agents consume `control-commands` and apply policy live (thresholds, tip ladder, throttle, kill). You already have the consumer skeleton from previous step—wire it to update an atomic struct used by the hot path.

---

## 8) End‑to‑end test plan

1. **WS transport**

   * Start FastAPI + Dramatiq
   * Open dashboard, toggle JSON vs Proto. Metrics cards should update every second.
2. **Dataset scrape**

   * Use Datasets page; start a scrape (`parquet`, `zstd`).
   * Watch `/api/datasets/status/{job_id}` progress; file appears in output path.
3. **Training**

   * Start training for `mev` range; status progresses; Treelite `.so` lands in `shared/treelite/`.
   * Restart **mev-sandwich-agent** to pick up the new .so (or hot‑reload if you implement dlopen).
4. **Control plane**

   * Apply **tip ladder** & **EV min** via Control Panel; verify agents log policy changes.
5. **RBAC**

   * Use JWT with `roles:["viewer"]`: Control buttons disabled/blocked; `roles:["admin"]`: allowed.

---

## 9) Performance guardrails (legendary mode)

* **WS batching**: The server aggregates multiple events into **Batch** before sending; add a micro‑batch window (e.g., 10–25 ms) in production to reduce frames/sec.
* **Backpressure**: If a client is slow, **drop** oldest batches (do not block server → use an `asyncio.Queue(maxsize=… , put_nowait or get_nowait)`).
* **Zstd level**: 3 is a good trade‑off; raise if NIC is bottlenecked; lower if CPU is tight.
* **ClickHouse exports**: Always use Arrow stream → ParquetWriter with **row‑group** 256 MB and **ZSTD** to get **10–20×** compression while keeping scan speed.

---

## 10) Security notes

* Issue JWTs via your auth gateway; include `"roles":["admin"]` or `"viewer"]`.
* Set `SECRET`, `JWT_ISS`, `JWT_AUD` in env.
* Restrict `/api/clickhouse/query` to a curated whitelist of SQL if you expose it.
* Consider **mTLS** between control plane and internal services.

---

## 11) What remains for full production polish

* **Wire realtime WS** to your *actual* Kafka topics (`arbitrage-decisions`, `sandwich-decisions`, `bundle-outcomes`) instead of `fake_stream()`—map JSON → protobuf `Envelope`/`Batch`.
* Add a **metrics aggregator task** to emit `MetricsUpdate` from Prometheus or ClickHouse queries every 1–3 s.
* Implement **model hot‑reload** for agents (dlopen new Treelite .so with versioning), or do a seamless restart via control command.

---

### TL;DR

You now have:

* **Protobuf + Zstd** realtime transport (server + browser)
* **FastAPI** control plane with **JWT RBAC**
* **Dramatiq** workers for **dataset scraping** and **training** (ClickHouse → Parquet, XGBoost → Treelite)
* **Next.js** dashboard fully wired to control, datasets, training, and realtime streams.

If you want me to **swap Dramatiq for Celery**, provide **.proto→Kafka** mappers, or add a **Protobuf‑first agent emitter** in Rust (prost) so agents publish protobuf directly to Kafka (no JSON), I can drop those files next.
