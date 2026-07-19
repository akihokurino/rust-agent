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

// ---- Tool トレイトの既定実装 ----

#[test]
fn spec_is_what_bedrock_expects() {
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
fn defaults_are_inert() {
    // 通常のツールは LLM を消費しないので 0、タイムアウトは Agent の既定に委ねる
    assert_eq!(Minimal.sub_agent_usage(), (0, 0));
    assert_eq!(Minimal.timeout(), None);
}

// ---- AgentTool ----

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
async fn agent_tool_exposes_only_a_prompt() {
    let t = agent_tool().await;

    assert_eq!(t.name(), "research");
    assert_eq!(t.description(), "詳しく調べる");

    // LLM にエージェントを生成させる手段を渡さないため、入力は prompt だけに絞る
    let schema = t.input_schema();
    assert_eq!(schema["required"], json!(["prompt"]));
    assert_eq!(
        schema["properties"].as_object().unwrap().keys().len(),
        1,
        "prompt 以外の入力が生えている"
    );
}

#[tokio::test]
async fn agent_tool_rejects_missing_prompt() {
    let t = agent_tool().await;

    // prompt を検証してから run するので、ここで Bedrock は呼ばれない
    let e = t.execute(json!({})).await.unwrap_err();
    assert_eq!(e.kind, Kind::ValidationException);

    let e = t.execute(json!({ "prompt": 42 })).await.unwrap_err();
    assert_eq!(e.kind, Kind::ValidationException);
}

#[tokio::test]
async fn agent_tool_usage_starts_at_zero() {
    assert_eq!(agent_tool().await.sub_agent_usage(), (0, 0));
}

// ---- FetchUrl / WebSearch ----

#[test]
fn http_tools_declare_their_own_timeout() {
    // Agent 既定の 60 秒より短く抑えて、固まったときの待ちを減らす
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
    // url が無い / 型が違う
    assert!(FetchUrl.execute(json!({})).await.is_err());
    assert!(FetchUrl.execute(json!({ "url": 1 })).await.is_err());
    // URL として解釈できない
    assert_eq!(
        FetchUrl
            .execute(json!({ "url": "not a url" }))
            .await
            .unwrap_err()
            .kind,
        Kind::ValidationException
    );
}

/// LLM が URL を自由に組み立てられる以上、内部ネットワークへの到達は
/// ツール側で塞ぐしかない（SSRF）
#[tokio::test]
async fn fetch_url_blocks_internal_addresses() {
    let cases = [
        // EC2 / ECS のインスタンスメタデータ。IAM 認証情報が漏れる経路
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        // Alibaba のメタデータ (CGNAT 帯)
        "http://100.100.100.200/",
        "http://127.0.0.1:8080/",
        "http://localhost:8080/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
        // IPv6 リテラル。host_str() が "[::1]" と括弧付きで返るので取りこぼしやすい
        "http://[::1]/",
        // IPv4 射影アドレスによる迂回
        "http://[::ffff:169.254.169.254]/",
        // http/https 以外
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

/// 外部ネットワークに出るので既定では実行しない。
/// `cargo test --test tools -- --ignored` で実行する
#[tokio::test]
#[ignore]
async fn fetch_url_follows_redirects_and_strips_markup() {
    let res = FetchUrl
        .execute(json!({ "url": "https://www.cybozu.co.jp/" }))
        .await
        .unwrap();

    // 301 を自前で追いかけ、ホップごとに検証をやり直している
    assert_eq!(res["status"], 200);
    assert_ne!(res["url"], "https://www.cybozu.co.jp/");

    let content = res["content"].as_str().unwrap();
    assert!(content.contains("サイボウズ"));
    assert!(!content.contains("<"), "タグが残っている");
    assert!(!content.contains("function"), "script の中身が残っている");
}
