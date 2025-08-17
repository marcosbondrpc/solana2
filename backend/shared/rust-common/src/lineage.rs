use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DecisionLineage<'a> {
    pub ts_ns: i128,
    pub request_id: &'a str,
    pub agent_id: &'a str,
    pub model_version: &'a str,
    pub slot: u64,
    pub leader: &'a str,
    pub route: &'a str,
    pub tip_lamports: u64,
    pub lamports_per_cu: f64,
    pub expected_cu: f64,
    pub ev_estimate: f64,
    pub p_land_prior: f64,
    pub decision_ms: f64,
    pub submit_ms: Option<f64>,
    pub land_ms: Option<f64>,
    pub tx_signature: &'a str,
    pub bundle_hash_b64: &'a str,
    pub jito_code: Option<&'a str>,
    pub dedupe_key_b3_hex: &'a str,
    pub features_json: &'a str,
    pub control_actor: Option<&'a str>,
    pub node_id: &'a str,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Counterfactual<'a> {
    pub ts_ns: i128,
    pub request_id: &'a str,
    pub slot: u64,
    pub leader: &'a str,
    pub chosen_route: &'a str,
    pub shadow_route: &'a str,
    pub predicted_land_prob: f64,
    pub predicted_ev: f64,
    pub realized_land: Option<u8>,
    pub realized_ev: Option<f64>,
}