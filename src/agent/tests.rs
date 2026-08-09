use super::*;
use crate::types::agent::{InvokeResult, Role, Usage};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

const MODEL: Model = Model::BedrockClaudeSonnet46;

#[derive(Default)]
struct Recorder {
    calls: AtomicU32,
    last_history: Mutex<Vec<Message>>,
    tool_choices: Mutex<Vec<String>>,
}

struct FakeLlm {
    script_repeating_last: Mutex<VecDeque<InvokeResult>>,
    rec: Arc<Recorder>,
    delay: Duration,
}

#[async_trait]
impl llm::LLM for FakeLlm {
    async fn invoke(
        &self,
        _: &Model,
        _: &str,
        _: u32,
        messages: &[Message],
        _: &[&dyn tool::Tool],
        tool_choice: &ToolChoice,
    ) -> Result<InvokeResult, AgentError> {
        tokio::time::sleep(self.delay).await;
        self.rec.calls.fetch_add(1, Ordering::Relaxed);
        *self.rec.last_history.lock().unwrap() = messages.to_vec();
        self.rec
            .tool_choices
            .lock()
            .unwrap()
            .push(match tool_choice {
                ToolChoice::Auto => "auto".into(),
                ToolChoice::Specific(n) => n.clone(),
            });

        let mut script = self.script_repeating_last.lock().unwrap();
        Ok(if script.len() > 1 {
            script.pop_front().unwrap()
        } else {
            script.front().unwrap().clone()
        })
    }
}

fn ends(text: &str, usage: (u32, u32)) -> InvokeResult {
    InvokeResult {
        content: vec![ResultBlock::Text { text: text.into() }],
        stop_reason: StopReason::EndTurn,
        usage: Usage {
            input_tokens: usage.0,
            output_tokens: usage.1,
        },
    }
}

fn calls(names: &[&str], usage: (u32, u32)) -> InvokeResult {
    InvokeResult {
        content: names
            .iter()
            .enumerate()
            .map(|(i, name)| ResultBlock::ToolUse {
                id: format!("tu_{i}"),
                name: (*name).into(),
                input: json!({ "prompt": "go" }),
            })
            .collect(),
        stop_reason: StopReason::ToolUse,
        usage: Usage {
            input_tokens: usage.0,
            output_tokens: usage.1,
        },
    }
}

fn responds(input: Value) -> InvokeResult {
    InvokeResult {
        content: vec![ResultBlock::ToolUse {
            id: "r".into(),
            name: "respond".into(),
            input,
        }],
        stop_reason: StopReason::ToolUse,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
        },
    }
}

fn agent_with(
    script: Vec<InvokeResult>,
    tools: Vec<Box<dyn tool::Tool>>,
) -> (Agent, Arc<Recorder>) {
    let rec = Arc::new(Recorder::default());
    let mut providers: HashMap<Model, Box<dyn llm::LLM>> = HashMap::new();
    providers.insert(
        MODEL,
        Box::new(FakeLlm {
            script_repeating_last: Mutex::new(script.into()),
            rec: rec.clone(),
            delay: Duration::ZERO,
        }),
    );

    let agent = Agent {
        system_prompt: String::new(),
        max_turns: MAX_TURNS,
        max_tokens: 1024,
        max_total_tokens: u32::MAX,
        default_tool_timeout: Duration::from_secs(60),
        tools,
        providers,
    };
    (agent, rec)
}

fn go() -> Vec<Input> {
    vec![Input::Text("go".into())]
}

struct SlowTool {
    name: &'static str,
    delay: Duration,
}
#[async_trait]
impl tool::Tool for SlowTool {
    fn name(&self) -> String {
        self.name.into()
    }
    fn description(&self) -> String {
        "test".into()
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn execute(&self, _: Value) -> Result<Value, AgentError> {
        tokio::time::sleep(self.delay).await;
        Ok(json!("done"))
    }
}

fn instant(name: &'static str) -> Box<SlowTool> {
    Box::new(SlowTool {
        name,
        delay: Duration::ZERO,
    })
}

#[tokio::test]
async fn max_turns_stops_an_llm_that_keeps_calling_tools() {
    let (agent, rec) = agent_with(vec![calls(&["noop"], (1, 1))], vec![instant("noop")]);

    let err = agent.run(&MODEL, go(), None).await.unwrap_err();

    assert_eq!(err.kind, Kind::MaxTurnsExceeded);
    assert_eq!(rec.calls.load(Ordering::Relaxed), MAX_TURNS);
}

#[tokio::test]
async fn token_budget_stops_the_loop_before_max_turns_does() {
    let (mut agent, rec) = agent_with(vec![calls(&["noop"], (100, 50))], vec![instant("noop")]);
    agent.max_total_tokens = 300;

    let err = agent.run(&MODEL, go(), None).await.unwrap_err();

    assert_eq!(err.kind, Kind::TokenBudgetExceeded);
    assert_eq!(rec.calls.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn an_exhausted_budget_still_returns_an_answer_that_was_reached() {
    let (mut agent, _) = agent_with(vec![ends("答え", (1000, 1000))], vec![]);
    agent.max_total_tokens = 10;

    let res = agent.run(&MODEL, go(), None).await.unwrap();

    assert_eq!(res.content, "答え");
    assert_eq!(res.input_tokens + res.output_tokens, 2000);
}

#[tokio::test]
async fn calling_an_unregistered_tool_is_an_error() {
    let (agent, _) = agent_with(vec![calls(&["nope"], (1, 1))], vec![]);

    assert_eq!(
        agent.run(&MODEL, go(), None).await.unwrap_err().kind,
        Kind::ToolNotFound
    );
}

#[tokio::test]
async fn a_sub_agents_tokens_are_included_in_the_parent_result() {
    let (child, _) = child_agent(ends("child done", (700, 300)), Duration::ZERO);

    let (parent, _) = agent_with(
        vec![calls(&["research"], (10, 5)), ends("done", (1, 2))],
        vec![Box::new(tool::AgentTool::new(
            "research", "test", MODEL, child,
        ))],
    );

    let res = parent.run(&MODEL, go(), None).await.unwrap();

    assert_eq!(res.input_tokens, 10 + 700 + 1);
    assert_eq!(res.output_tokens, 5 + 300 + 2);
}

/// 好きな応答と 1 回あたりの所要時間を持つ子エージェント
fn child_agent(response: InvokeResult, per_call: Duration) -> (Agent, Arc<Recorder>) {
    let (mut agent, _) = agent_with(vec![], vec![instant("noop")]);
    let rec = Arc::new(Recorder::default());
    let mut providers: HashMap<Model, Box<dyn llm::LLM>> = HashMap::new();
    providers.insert(
        MODEL,
        Box::new(FakeLlm {
            script_repeating_last: Mutex::new(vec![response].into()),
            rec: rec.clone(),
            delay: per_call,
        }),
    );
    agent.providers = providers;
    agent.max_total_tokens = u32::MAX;
    (agent, rec)
}

#[tokio::test]
async fn a_sub_agent_cannot_outspend_the_parent_budget() {
    // 子の予算は無制限に設定してある
    let (child, child_rec) = child_agent(calls(&["noop"], (100, 100)), Duration::ZERO);

    let (mut parent, _) = agent_with(
        vec![calls(&["research"], (10, 10)), ends("done", (1, 1))],
        vec![Box::new(tool::AgentTool::new(
            "research", "test", MODEL, child,
        ))],
    );
    parent.max_total_tokens = 500;

    let err = parent.run(&MODEL, go(), None).await.unwrap_err();
    assert_eq!(err.kind, Kind::TokenBudgetExceeded);

    // 子は自分の上限ではなく、親の予算で止まっている
    let child_spent = child_rec.calls.load(Ordering::Relaxed) * 200;
    assert!(child_spent <= 500 + 200, "子が {child_spent} 使った");
}

#[tokio::test(start_paused = true)]
async fn tokens_spent_by_a_sub_agent_survive_its_cancellation() {
    // 子は 1 回 10 秒かけて 50,000 使う。親は 25 秒で打ち切る
    let (child, _) = child_agent(calls(&["noop"], (50_000, 0)), Duration::from_secs(10));

    let (mut parent, _) = agent_with(
        vec![calls(&["research"], (1, 1)), ends("done", (1, 1))],
        vec![Box::new(tool::AgentTool::new(
            "research", "test", MODEL, child,
        ))],
    );
    parent.default_tool_timeout = Duration::from_secs(25);

    let res = parent.run(&MODEL, go(), None).await.unwrap();

    // 打ち切りまでに完了した 2 回分が計上されていること
    assert_eq!(res.input_tokens, 1 + 100_000 + 1, "{res:?}");
}

#[tokio::test]
async fn tool_calls_beyond_the_per_turn_cap_are_not_run() {
    let requested = MAX_TOOL_CALLS_PER_TURN as usize + 3;
    let names = vec!["sub"; requested];
    let (agent, rec) = agent_with(
        vec![calls(&names, (1, 1)), ends("done", (1, 1))],
        vec![instant("sub")],
    );

    agent.run(&MODEL, go(), None).await.unwrap();

    let history = rec.last_history.lock().unwrap();
    let blocks = &history.last().unwrap().content;
    assert_eq!(blocks.len(), requested);
    let errors = blocks
        .iter()
        .filter(|b| {
            matches!(b, MessageBlock::ToolResult { is_error, content, .. }
                    if *is_error && content.contains("TooManyToolCalls"))
        })
        .count();
    assert_eq!(errors, 3);
}

#[tokio::test(start_paused = true)]
async fn three_tools_in_one_turn_finish_in_the_time_of_one() {
    let slow = |name| {
        Box::new(SlowTool {
            name,
            delay: Duration::from_secs(3),
        })
    };
    let (agent, _) = agent_with(
        vec![calls(&["a", "b", "c"], (1, 1)), ends("done", (1, 1))],
        vec![slow("a"), slow("b"), slow("c")],
    );

    let start = tokio::time::Instant::now();
    agent.run(&MODEL, go(), None).await.unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed >= Duration::from_secs(3), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(4), "{elapsed:?}");
}

#[tokio::test(start_paused = true)]
async fn a_hung_tool_times_out_and_comes_back_to_the_llm_as_an_error() {
    let (mut agent, rec) = agent_with(
        vec![calls(&["hang"], (1, 1)), ends("諦めます", (1, 1))],
        vec![Box::new(SlowTool {
            name: "hang",
            delay: Duration::from_secs(9999),
        })],
    );
    agent.default_tool_timeout = Duration::from_millis(50);

    let res = agent.run(&MODEL, go(), None).await.unwrap();
    assert_eq!(res.content, "諦めます");

    let history = rec.last_history.lock().unwrap();
    match &history.last().unwrap().content[0] {
        MessageBlock::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("ToolTimeout"), "{content}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn a_tool_timeout_overrides_the_agent_default() {
    struct Impatient;
    #[async_trait]
    impl tool::Tool for Impatient {
        fn name(&self) -> String {
            "hang".into()
        }
        fn description(&self) -> String {
            "test".into()
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        fn timeout(&self) -> Option<Duration> {
            Some(Duration::from_millis(10))
        }
        async fn execute(&self, _: Value) -> Result<Value, AgentError> {
            tokio::time::sleep(Duration::from_secs(9999)).await;
            Ok(json!("never"))
        }
    }

    let (mut agent, _) = agent_with(
        vec![calls(&["hang"], (1, 1)), ends("done", (1, 1))],
        vec![Box::new(Impatient)],
    );
    agent.default_tool_timeout = Duration::from_secs(600);

    let start = tokio::time::Instant::now();
    agent.run(&MODEL, go(), None).await.unwrap();

    assert!(
        start.elapsed() < Duration::from_secs(1),
        "{:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn a_plain_text_reply_makes_the_next_turn_name_respond() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct Answer {
        pref: String,
    }

    let (agent, rec) = agent_with(
        vec![
            ends("東京です", (1, 1)),
            responds(json!({ "pref": "東京" })),
        ],
        vec![],
    );

    let res = agent.run_typed::<Answer>(&MODEL, go(), None).await.unwrap();

    assert_eq!(res.content.pref, "東京");
    assert_eq!(*rec.tool_choices.lock().unwrap(), ["auto", "respond"]);
}

#[tokio::test]
async fn structured_output_that_does_not_match_the_schema_is_an_error() {
    #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    struct Answer {
        #[allow(dead_code)]
        pref: String,
    }

    let (agent, _) = agent_with(vec![responds(json!({ "pref": 42 }))], vec![]);

    assert_eq!(
        agent
            .run_typed::<Answer>(&MODEL, go(), None)
            .await
            .unwrap_err()
            .kind,
        Kind::UnknownException
    );
}

#[tokio::test]
async fn passed_in_history_precedes_the_new_message() {
    let (agent, rec) = agent_with(vec![ends("答え", (1, 1))], vec![]);

    let prior = vec![
        Message::user(vec![MessageBlock::Text {
            text: "前の質問".into(),
        }]),
        Message {
            role: Role::Assistant,
            content: vec![MessageBlock::Text {
                text: "前の答え".into(),
            }],
        },
    ];

    agent.run(&MODEL, go(), Some(prior)).await.unwrap();

    let seen = rec.last_history.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].role, Role::User);
    assert_eq!(seen[1].role, Role::Assistant);
    assert_eq!(
        seen[2].content,
        go().into_iter().map(Into::into).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn the_returned_history_carries_the_whole_conversation() {
    let (agent, _) = agent_with(vec![ends("答え", (1, 1))], vec![]);

    let res = agent.run(&MODEL, go(), None).await.unwrap();

    assert_eq!(res.history.len(), 2);
    assert_eq!(res.history[0].role, Role::User);
    assert_eq!(res.history[1].role, Role::Assistant);
}

#[tokio::test]
async fn a_structured_answer_is_not_left_in_the_returned_history() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct Answer {
        pref: String,
    }

    let (agent, _) = agent_with(vec![responds(json!({ "pref": "東京" }))], vec![]);

    let res = agent.run_typed::<Answer>(&MODEL, go(), None).await.unwrap();

    assert_eq!(res.content.pref, "東京");
    let dangling = res
        .history
        .iter()
        .flat_map(|m| &m.content)
        .any(|b| matches!(b, MessageBlock::ToolUse { name, .. } if name == "respond"));
    assert!(
        !dangling,
        "respond の tool_use が履歴に残っている: {:?}",
        res.history
    );
}

#[tokio::test]
async fn a_custom_max_turns_is_respected() {
    let (mut agent, rec) = agent_with(vec![calls(&["noop"], (1, 1))], vec![instant("noop")]);
    agent.max_turns = 3;

    let err = agent.run(&MODEL, go(), None).await.unwrap_err();

    assert_eq!(err.kind, Kind::MaxTurnsExceeded);
    assert_eq!(rec.calls.load(Ordering::Relaxed), 3);
}
