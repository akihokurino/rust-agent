use rust_agent::{Agent, Kind, Model};
use std::time::Duration;

#[tokio::test]
async fn defaults() {
    let a = Agent::builder().build().await.unwrap();

    assert_eq!(a.system_prompt, "");
    assert_eq!(a.max_tokens, 1024);
    assert_eq!(a.max_turns, 10);
    assert_eq!(a.max_total_tokens, 500_000);
    assert_eq!(a.max_tool_calls_per_turn, 8);
    assert_eq!(a.default_tool_timeout, Duration::from_secs(60));
    assert!(a.tools.is_empty());
}

#[tokio::test]
async fn overrides_are_applied() {
    let a = Agent::builder()
        .system_prompt("あなたは調査エージェントです")
        .max_tokens(4096)
        .max_turns(3)
        .max_total_tokens(50_000)
        .max_tool_calls_per_turn(3)
        .default_tool_timeout(Duration::from_secs(5))
        .use_models(vec![Model::BedrockClaudeSonnet46])
        .build()
        .await
        .unwrap();

    assert_eq!(a.system_prompt, "あなたは調査エージェントです");
    assert_eq!(a.max_tokens, 4096);
    assert_eq!(a.max_turns, 3);
    assert_eq!(a.max_total_tokens, 50_000);
    assert_eq!(a.max_tool_calls_per_turn, 3);
    assert_eq!(a.default_tool_timeout, Duration::from_secs(5));
}

#[tokio::test]
async fn building_without_a_model_is_rejected() {
    match Agent::builder().use_models(vec![]).build().await {
        Err(e) => assert_eq!(e.kind, Kind::ValidationException),
        Ok(_) => panic!("モデル指定なしで build できてしまった"),
    }
}

#[tokio::test]
async fn tools_keep_their_registration_order() {
    let a = Agent::builder()
        .add_tool(Box::new(tools::Named("a")))
        .add_tool(Box::new(tools::Named("b")))
        .build()
        .await
        .unwrap();

    let names: Vec<String> = a.tools.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn model_id_is_the_bedrock_inference_profile() {
    assert_eq!(
        Model::BedrockClaudeSonnet46.to_string(),
        "jp.anthropic.claude-sonnet-4-6"
    );
}

mod tools {
    use async_trait::async_trait;
    use rust_agent::{AgentError, Tool};
    use serde_json::{Value, json};

    pub struct Named(pub &'static str);

    #[async_trait]
    impl Tool for Named {
        fn name(&self) -> String {
            self.0.into()
        }
        fn description(&self) -> String {
            "test".into()
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn execute(&self, _: Value) -> Result<Value, AgentError> {
            Ok(json!(null))
        }
    }
}
