use super::*;

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
async fn exposes_a_prompt_and_nothing_else() {
    let t = agent_tool().await;

    assert_eq!(t.name(), "research");
    assert_eq!(t.description(), "詳しく調べる");

    let schema = t.input_schema();
    assert_eq!(schema["required"], json!(["prompt"]));
    assert_eq!(schema["properties"].as_object().unwrap().keys().len(), 1);
}

#[tokio::test]
async fn rejects_a_missing_or_non_string_prompt() {
    let t = agent_tool().await;

    assert_eq!(
        t.execute(json!({})).await.unwrap_err().kind,
        Kind::ValidationException
    );
    assert_eq!(
        t.execute(json!({ "prompt": 42 })).await.unwrap_err().kind,
        Kind::ValidationException
    );
}
