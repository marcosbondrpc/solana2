use clickhouse::{Client, Row};
use serde::Serialize;
use crate::{DecisionLineage, Counterfactual};
use chrono::TimeZone;

#[derive(Clone)]
pub struct ChSink {
    client: Client,
}

impl ChSink {
    pub fn new(url: &str, db: &str) -> Self {
        let client = Client::default().with_url(url).with_database(db);
        Self { client }
    }

    pub async fn insert_lineage(&self, rows: &[DecisionLineage<'_>]) -> clickhouse::error::Result<()> {
        #[derive(Row, Serialize)]
        struct R<'a> {
            ts: chrono::DateTime<chrono::Utc>,
            request_id: &'a str,
            agent_id: &'a str,
            model_version: &'a str,
            slot: u64,
            leader: &'a str,
            route: &'a str,
            tip_lamports: u64,
            lamports_per_cu: f64,
            expected_cu: f64,
            ev_estimate: f64,
            p_land_prior: f64,
            decision_ms: f64,
            submit_ms: Option<f64>,
            land_ms: Option<f64>,
            tx_signature: &'a str,
            bundle_hash: &'a str,
            jito_code: Option<&'a str>,
            dedupe_key_b3: &'a str,
            features: &'a str,
            control_actor: Option<&'a str>,
            node_id: &'a str,
        }
        let mut insert = self.client.insert("mev_decision_lineage")?;
        for r in rows {
            let row = R {
                ts: chrono::Utc.timestamp_nanos(r.ts_ns as i64),
                request_id: r.request_id, agent_id: r.agent_id, model_version: r.model_version,
                slot: r.slot, leader: r.leader, route: r.route, tip_lamports: r.tip_lamports,
                lamports_per_cu: r.lamports_per_cu, expected_cu: r.expected_cu,
                ev_estimate: r.ev_estimate, p_land_prior: r.p_land_prior,
                decision_ms: r.decision_ms, submit_ms: r.submit_ms, land_ms: r.land_ms,
                tx_signature: r.tx_signature, bundle_hash: r.bundle_hash_b64,
                jito_code: r.jito_code, dedupe_key_b3: r.dedupe_key_b3_hex,
                features: r.features_json, control_actor: r.control_actor, node_id: r.node_id,
            };
            insert.write(&row).await?;
        }
        insert.end().await?;
        Ok(())
    }

    pub async fn insert_counterfactuals(&self, rows: &[Counterfactual<'_>]) -> clickhouse::error::Result<()> {
        #[derive(Row, Serialize)]
        struct R<'a> {
            ts: chrono::DateTime<chrono::Utc>,
            request_id: &'a str,
            slot: u64,
            leader: &'a str,
            chosen_route: &'a str,
            shadow_route: &'a str,
            predicted_land_prob: f64,
            predicted_ev: f64,
            realized_land: Option<u8>,
            realized_ev: Option<f64>,
        }
        
        let mut insert = self.client.insert("mev_counterfactuals")?;
        for r in rows {
            let row = R {
                ts: chrono::Utc.timestamp_nanos(r.ts_ns as i64),
                request_id: r.request_id,
                slot: r.slot,
                leader: r.leader,
                chosen_route: r.chosen_route,
                shadow_route: r.shadow_route,
                predicted_land_prob: r.predicted_land_prob,
                predicted_ev: r.predicted_ev,
                realized_land: r.realized_land,
                realized_ev: r.realized_ev,
            };
            insert.write(&row).await?;
        }
        insert.end().await?;
        Ok(())
    }
}