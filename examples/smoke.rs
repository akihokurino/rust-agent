use async_trait::async_trait;
use rust_agent::{Agent, AgentError, Kind, Model, Tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize, JsonSchema)]
struct WebSearchInput {
    query: String,
}

struct WebSearch;

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> String {
        "web_search".into()
    }

    fn description(&self) -> String {
        "Googleで検索して上位の検索結果を返す".into()
    }

    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(WebSearchInput)).unwrap()
    }

    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: WebSearchInput = serde_json::from_value(input)?;
        let api_key = std::env::var("SERPER_API_KEY").unwrap_or_default();

        let resp = reqwest::Client::new()
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .json(&json!({ "q": args.query, "gl": "jp", "hl": "ja", "num": 5 }))
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?
            .json::<Value>()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;

        Ok(json!({ "results": resp["organic"] }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let agent = Agent::builder()
        .system_prompt("必要に応じて web_search ツールで調べ、日本語で簡潔に答えてください。")
        .add_tool(Box::new(WebSearch))
        .build()
        .await?;

    let out = agent
        .run(&Model::BedrockClaudeSonnet46, "現在の日本の総理大臣は誰？")
        .await?;

    println!("{:?}", out);
    Ok(())
}
