use clickhouse::{Client, Row};

#[derive(Row)]
pub struct DecisionRow<'a> {
    pub slot: u64,
    pub arm: &'a str,
    pub p_land: f64,
    pub ev: i64,
    pub tip: u64,
    pub pool: &'a str,
    pub ts_ns: i128,
    pub counterfactual: bool,
}

pub async fn write_decision(client: &Client, row: DecisionRow<'_>) -> clickhouse::Result<()> {
    client.insert("mev_decision_lineage").write(&row).await
}



