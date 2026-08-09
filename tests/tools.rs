use async_trait::async_trait;
use rust_agent::{AgentError, FetchUrl, Kind, Tool, WebSearch};
use serde_json::{Value, json};
use std::time::Duration;

struct Minimal;

#[async_trait]
impl Tool for Minimal {
    fn name(&self) -> &str {
        "minimal"
    }
    fn description(&self) -> &str {
        "説明"
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
fn a_plain_tool_has_no_timeout_of_its_own() {
    assert_eq!(Minimal.timeout(), None);
}

#[test]
fn http_tools_cap_themselves_below_the_agent_default() {
    assert_eq!(FetchUrl.timeout(), Some(Duration::from_secs(30)));
    assert_eq!(
        WebSearch {
            serper_api_key: None
        }
        .timeout(),
        Some(Duration::from_secs(30))
    );
}

#[test]
fn http_tools_have_schemas() {
    assert_eq!(FetchUrl.name(), "fetch_url");
    assert_eq!(
        FetchUrl.input_schema()["properties"]["url"]["type"],
        "string"
    );

    assert_eq!(
        WebSearch {
            serper_api_key: None
        }
        .name(),
        "web_search"
    );
    assert_eq!(
        WebSearch {
            serper_api_key: None
        }
        .input_schema()["properties"]["query"]["type"],
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
