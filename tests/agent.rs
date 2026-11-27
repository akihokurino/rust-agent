#[path = "agent/bedrock.rs"]
mod bedrock;

use async_trait::async_trait;
use bedrock::{Response, exercise};
use rust_agent::{Agent, AgentBuilder, AgentError, Input, Kind, Message, MessageBlock, Role, Tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use std::future::pending;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::timeout;

const MODEL: rust_agent::Model = rust_agent::Model::BedrockClaudeSonnet46;

fn builder() -> AgentBuilder {
    Agent::builder()
        .use_models(vec![MODEL])
        .system_prompt("Test assistant")
        .max_tokens(123)
}

fn input() -> Vec<Input> {
    vec![Input::Text("go".into())]
}

fn response(stop_reason: &str, content: Value, tokens: (u32, u32)) -> Response {
    Response::Ok(json!({
        "content": content,
        "stop_reason": stop_reason,
        "usage": { "input_tokens": tokens.0, "output_tokens": tokens.1 },
    }))
}

fn call(id: &str, name: &str, input: Value) -> Value {
    json!({ "type": "tool_use", "id": id, "name": name, "input": input })
}

struct RecordingTool {
    name: &'static str,
    calls: Arc<Mutex<Vec<Value>>>,
    result: Result<Value, AgentError>,
}

#[async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Local fixture tool"
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        self.calls.lock().unwrap().push(input);
        self.result.clone()
    }
}

#[tokio::test]
async fn routes_tool_calls_and_replays_success_and_failure() {
    let replies = [
        response(
            "tool_use",
            json!([
                call("call_lookup", "lookup", json!({"id": 7})),
                call("call_reject", "reject", json!({"id": 8})),
            ]),
            (11, 3),
        ),
        response(
            "end_turn",
            json!([{ "type": "text", "text": "done" }]),
            (17, 4),
        ),
    ];
    let requests = exercise(
        "routes_tool_calls_and_replays_success_and_failure",
        replies,
        async {
            let success = Arc::new(Mutex::new(Vec::new()));
            let failure = Arc::new(Mutex::new(Vec::new()));
            let unused = Arc::new(Mutex::new(Vec::new()));
            let agent = builder()
                .add_tool(Box::new(RecordingTool {
                    name: "lookup",
                    calls: Arc::clone(&success),
                    result: Ok(json!({"price": 42})),
                }))
                .add_tool(Box::new(RecordingTool {
                    name: "reject",
                    calls: Arc::clone(&failure),
                    result: Err(Kind::ValidationException.with("invalid id")),
                }))
                .add_tool(Box::new(RecordingTool {
                    name: "unused",
                    calls: Arc::clone(&unused),
                    result: Ok(Value::Null),
                }))
                .build()
                .await
                .unwrap();
            let previous = Message::user(vec![MessageBlock::Text {
                text: "existing history".into(),
            }]);
            let answer = agent
                .run(&MODEL, input(), Some(vec![previous]))
                .await
                .unwrap();

            assert_eq!(&*success.lock().unwrap(), &[json!({"id": 7})]);
            assert_eq!(&*failure.lock().unwrap(), &[json!({"id": 8})]);
            assert!(unused.lock().unwrap().is_empty());
            assert_eq!(answer.content, "done");
            assert_eq!((answer.input_tokens, answer.output_tokens), (28, 7));
            assert_eq!(answer.history.last().unwrap().role, Role::Assistant);
        },
    )
    .await;
    if let Some(requests) = requests {
        assert_eq!(requests[0]["system"], "Test assistant");
        assert_eq!(requests[0]["max_tokens"], 123);
        assert_eq!(requests[0]["tools"][0]["name"], "lookup");
        assert_eq!(requests[0]["tools"][1]["name"], "reject");
        assert_eq!(
            requests[0]["messages"][0]["content"][0]["text"],
            "existing history"
        );
        assert_eq!(requests[0]["messages"][1]["content"][0]["text"], "go");

        let results = requests[1]["messages"].as_array().unwrap().last().unwrap();
        assert_eq!(results["role"], "user");
        let results = results["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0],
            json!({
                "type": "tool_result",
                "tool_use_id": "call_lookup",
                "content": r#"{"price":42}"#,
                "is_error": false,
            })
        );
        assert_eq!(results[1]["type"], "tool_result");
        assert_eq!(results[1]["tool_use_id"], "call_reject");
        assert_eq!(results[1]["is_error"], true);
        assert!(
            results[1]["content"]
                .as_str()
                .unwrap()
                .contains("invalid id")
        );
    }
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Answer {
    value: u32,
}

#[tokio::test]
async fn requests_structured_output_after_a_plain_reply() {
    let replies = [
        response(
            "end_turn",
            json!([{ "type": "text", "text": "I have an answer" }]),
            (5, 2),
        ),
        response(
            "tool_use",
            json!([call("answer", "respond", json!({"value": 42}))]),
            (10, 3),
        ),
    ];
    let requests = exercise(
        "requests_structured_output_after_a_plain_reply",
        replies,
        async {
            let agent = builder().build().await.unwrap();
            let answer = agent
                .run_typed::<Answer>(&MODEL, input(), None)
                .await
                .unwrap();

            assert_eq!(answer.content, Answer { value: 42 });
            assert_eq!(answer.history.len(), 2);
            assert_eq!((answer.input_tokens, answer.output_tokens), (15, 5));
        },
    )
    .await;
    if let Some(requests) = requests {
        assert_eq!(requests[0]["tools"][0]["name"], "respond");
        assert_eq!(
            requests[0]["tools"][0]["input_schema"]["properties"]["value"]["type"],
            "integer"
        );
        assert_eq!(
            requests[1]["tool_choice"],
            json!({ "type": "tool", "name": "respond" })
        );
    }
}

#[tokio::test]
async fn enforces_turn_and_token_limits_before_the_next_request() {
    let retry = || {
        response(
            "tool_use",
            json!([call("retry", "lookup", json!({}))]),
            (2, 1),
        )
    };
    exercise(
        "enforces_turn_and_token_limits_before_the_next_request",
        [retry(), retry()],
        async {
            for (builder, kind) in [
                (builder().max_turns(1), Kind::MaxTurnsExceeded),
                (builder().max_total_tokens(3), Kind::TokenBudgetExceeded),
            ] {
                let calls = Arc::new(Mutex::new(Vec::new()));
                let agent = builder
                    .add_tool(Box::new(RecordingTool {
                        name: "lookup",
                        calls: Arc::clone(&calls),
                        result: Ok(Value::Null),
                    }))
                    .build()
                    .await
                    .unwrap();

                assert_eq!(
                    agent.run(&MODEL, input(), None).await.unwrap_err().kind,
                    kind
                );
                assert_eq!(calls.lock().unwrap().len(), 1);
            }
        },
    )
    .await;
}

#[tokio::test]
async fn propagates_access_denied_without_running_tools() {
    exercise(
        "propagates_access_denied_without_running_tools",
        [Response::AccessDenied],
        async {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let agent = builder()
                .add_tool(Box::new(RecordingTool {
                    name: "lookup",
                    calls: Arc::clone(&calls),
                    result: Ok(Value::Null),
                }))
                .build()
                .await
                .unwrap();

            let error = agent.run(&MODEL, input(), None).await.unwrap_err();
            assert_eq!(error.kind, Kind::ModelAccessDeniedException);
            assert!(calls.lock().unwrap().is_empty());
        },
    )
    .await;
}

struct PendingTool {
    entered: Mutex<Option<oneshot::Sender<()>>>,
    dropped: Arc<AtomicBool>,
}

struct MarkDropped(Arc<AtomicBool>);

impl Drop for MarkDropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl Tool for PendingTool {
    fn name(&self) -> &str {
        "wait"
    }

    fn description(&self) -> &str {
        "Wait until cancelled"
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn execute(&self, _: Value) -> Result<Value, AgentError> {
        let _mark_dropped = MarkDropped(Arc::clone(&self.dropped));
        self.entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(())
            .unwrap();
        pending().await
    }
}

#[tokio::test]
async fn cancelling_the_run_drops_the_in_flight_tool() {
    let replies = [response(
        "tool_use",
        json!([call("wait", "wait", json!({}))]),
        (3, 1),
    )];
    exercise(
        "cancelling_the_run_drops_the_in_flight_tool",
        replies,
        async {
            let (signal, entered) = oneshot::channel();
            let dropped = Arc::new(AtomicBool::new(false));
            let agent: Agent = builder()
                .add_tool(Box::new(PendingTool {
                    entered: Mutex::new(Some(signal)),
                    dropped: Arc::clone(&dropped),
                }))
                .build()
                .await
                .unwrap();

            let run = tokio::spawn(async move { agent.run(&MODEL, input(), None).await });
            timeout(Duration::from_secs(5), entered)
                .await
                .unwrap()
                .unwrap();
            run.abort();
            assert!(run.await.unwrap_err().is_cancelled());
            assert!(dropped.load(Ordering::SeqCst));
        },
    )
    .await;
}
