use crate::agent::tool::Tool;
use crate::types::errors::{AgentError, Kind};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(serde::Deserialize, JsonSchema)]
struct WebSearchInput {
    query: String,
}
pub struct WebSearch {
    pub serper_api_key: Option<String>,
}
#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> String {
        "web_search".into()
    }
    fn description(&self) -> String {
        "Google検索を行い、検索結果を取得します。".into()
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(WebSearchInput)).unwrap()
    }
    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(30))
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: WebSearchInput = serde_json::from_value(input)?;
        let api_key = self
            .serper_api_key
            .as_deref()
            .ok_or_else(|| Kind::ValidationException.with("serper_api_key is required"))?;
        let resp = reqwest::Client::new()
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .json(&json!({ "q": args.query, "gl": "jp", "hl": "ja", "num": 5 }))
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?
            .error_for_status()
            .map_err(Kind::UnknownException.from_srcf())?
            .json::<Value>()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        Ok(json!({ "results": resp["organic"] }))
    }
}
