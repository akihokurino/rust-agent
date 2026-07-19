use async_trait::async_trait;
use rust_agent::{Agent, AgentError, AgentTool, FetchUrl, Kind, Model, Tool, WebSearch};
use serde_json::{Value, json};
use std::time::Duration;

struct Minimal;

#[async_trait]
impl Tool for Minimal {
    fn name(&self) -> String {
        "minimal".into()
    }
    fn description(&self) -> String {
        "説明".into()
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn execute(&self, _: Value) -> Result<Value, AgentError> {
        Ok(json!(null))
    }
}

#[test]
fn spec_is_the_shape_bedrock_expects() {
    assert_eq!(
        Minimal.spec(),
        json!({
            "name": "minimal",
            "description": "説明",
            "input_schema": { "type": "object" },
        })
    );
}

#[test]
fn a_plain_tool_reports_no_usage_and_no_timeout() {
    assert_eq!(Minimal.sub_agent_usage(), (0, 0));
    assert_eq!(Minimal.timeout(), None);
}

async fn agent_tool() -> AgentTool {
    let sub = Agent::builder().build().await.unwrap();
    AgentTool::new(
        "research",
        "詳しく調べる",
        Model::BedrockClaudeSonnet46,
        sub,
    )
}

#[tokio::test]
async fn agent_tool_exposes_a_prompt_and_nothing_else() {
    let t = agent_tool().await;

    assert_eq!(t.name(), "research");
    assert_eq!(t.description(), "詳しく調べる");

    let schema = t.input_schema();
    assert_eq!(schema["required"], json!(["prompt"]));
    assert_eq!(schema["properties"].as_object().unwrap().keys().len(), 1);
}

#[tokio::test]
async fn agent_tool_rejects_a_missing_or_non_string_prompt() {
    let t = agent_tool().await;

    let e = t.execute(json!({})).await.unwrap_err();
    assert_eq!(e.kind, Kind::ValidationException);

    let e = t.execute(json!({ "prompt": 42 })).await.unwrap_err();
    assert_eq!(e.kind, Kind::ValidationException);
}

#[tokio::test]
async fn agent_tool_usage_starts_at_zero() {
    assert_eq!(agent_tool().await.sub_agent_usage(), (0, 0));
}

#[test]
fn http_tools_cap_themselves_below_the_agent_default() {
    assert_eq!(FetchUrl.timeout(), Some(Duration::from_secs(30)));
    assert_eq!(WebSearch.timeout(), Some(Duration::from_secs(30)));
}

#[test]
fn http_tools_have_schemas() {
    assert_eq!(FetchUrl.name(), "fetch_url");
    assert_eq!(
        FetchUrl.input_schema()["properties"]["url"]["type"],
        "string"
    );

    assert_eq!(WebSearch.name(), "web_search");
    assert_eq!(
        WebSearch.input_schema()["properties"]["query"]["type"],
        "string"
    );
}

#[tokio::test]
async fn fetch_url_rejects_malformed_input() {
    assert!(FetchUrl.execute(json!({})).await.is_err());
    assert!(FetchUrl.execute(json!({ "url": 1 })).await.is_err());
    assert_eq!(
        FetchUrl
            .execute(json!({ "url": "not a url" }))
            .await
            .unwrap_err()
            .kind,
        Kind::ValidationException
    );
}

#[tokio::test]
async fn fetch_url_blocks_metadata_endpoints_and_private_networks() {
    let cases = [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://100.100.100.200/",
        "http://127.0.0.1:8080/",
        "http://localhost:8080/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:169.254.169.254]/",
        "file:///etc/passwd",
        "ftp://example.com/",
    ];

    for url in cases {
        let e = FetchUrl
            .execute(json!({ "url": url }))
            .await
            .expect_err(&format!("通ってしまった: {url}"));
        assert_eq!(e.kind, Kind::ValidationException, "{url}");
    }
}

#[tokio::test]
#[ignore]
async fn fetch_url_follows_redirects_and_strips_markup() {
    let res = FetchUrl
        .execute(json!({ "url": "https://www.cybozu.co.jp/" }))
        .await
        .unwrap();

    assert_eq!(res["status"], 200);
    assert_ne!(res["url"], "https://www.cybozu.co.jp/");

    let content = res["content"].as_str().unwrap();
    assert!(content.contains("サイボウズ"));
    assert!(!content.contains("<"));
    assert!(!content.contains("function"));
}
