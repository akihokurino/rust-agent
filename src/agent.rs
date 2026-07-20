use crate::agent::types::{Message, MessageBlock, ResultBlock, StopReason, ToolChoice};
use crate::llm::bedrock;
use crate::types::agent::AgentResult;
use crate::types::errors::AgentError;
use crate::types::errors::Kind::ValidationException;
use crate::types::model::Model;
use crate::{Input, Kind};
use futures::future::join_all;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

pub mod llm;
pub mod tool;
pub mod types;

/// 1 回の run に割り当てられた予算と、その消費を報告する先。
/// 報告先は親の `AgentTool` が持つカウンタで、1 ターンごとに書くことで
/// 途中で打ち切られても消費が失われないようにしている
struct Budget<'a> {
    limit: u32,
    sink: Option<(&'a AtomicU32, &'a AtomicU32)>,
}
impl Budget<'_> {
    fn record(&self, input: u32, output: u32) {
        if let Some((i, o)) = self.sink {
            i.fetch_add(input, Ordering::Relaxed);
            o.fetch_add(output, Ordering::Relaxed);
        }
    }
}

pub struct Agent {
    pub system_prompt: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    pub max_total_tokens: u32,
    pub max_tool_calls_per_turn: u32,
    pub default_tool_timeout: Duration,
    pub tools: Vec<Box<dyn tool::Tool>>,
    providers: HashMap<Model, Box<dyn llm::LLM>>,
}
impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    async fn loop_call<O>(
        &self,
        model: &Model,
        input: Vec<Input>,
        system: &str,
        tool_refs: &[&dyn tool::Tool],
        budget: Budget<'_>,
        finish: impl Fn(&types::InvokeResult) -> Option<Result<O, AgentError>>,
    ) -> Result<AgentResult<O>, AgentError> {
        // 指定モデルから利用する LLM アダプターを決定
        let llm = self.providers.get(model).ok_or(Kind::ModelNotConfigured)?;

        let tool_map = tool_refs
            .iter()
            .map(|t| (t.name(), t))
            .collect::<HashMap<_, _>>();

        let content: Vec<MessageBlock> = input.into_iter().map(Into::into).collect();
        let mut history = vec![Message::user(content)];
        let mut turns: u32 = 0;
        let mut input_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;
        let mut tool_choice = ToolChoice::Auto;

        // struct output を利用している場合は true となる
        let has_respond = tool_map.contains_key("respond");

        loop {
            // 異常時のために最大試行回数を決めておく
            if turns >= self.max_turns {
                return Err(Kind::MaxTurnsExceeded.default());
            }

            // ターン数だけでは、1 ターンあたりのツール呼び出し数やサブエージェントの深さで消費が積算されるため、トークン量でも上限を設ける
            // 完了した run は finish() で先に return するので、ここには来ない
            let spent = input_tokens.saturating_add(output_tokens);
            if spent >= budget.limit {
                return Err(Kind::TokenBudgetExceeded.with(format!(
                    "consumed {spent} tokens (budget: {})",
                    budget.limit
                )));
            }

            turns += 1;

            // LLM 実行
            let res = llm
                .invoke(
                    model,
                    system,
                    self.max_tokens,
                    &history,
                    tool_refs,
                    &tool_choice,
                )
                .await?;

            input_tokens += res.usage.input_tokens;
            output_tokens += res.usage.output_tokens;
            budget.record(res.usage.input_tokens, res.usage.output_tokens);

            // 完了条件を満たすか検証する
            // 満たしていた場合はそこで結果を返す
            if let Some(result) = finish(&res) {
                let content = result?;
                return Ok(AgentResult {
                    content,
                    input_tokens,
                    output_tokens,
                });
            }

            // respond 以外の通常のツールで実行リクエストが来ているものを収集
            let tool_calls: Vec<(String, String, serde_json::Value)> = res
                .content
                .iter()
                .filter_map(|b| match b {
                    ResultBlock::ToolUse { id, name, input } if name != "respond" => {
                        Some((id.clone(), name.clone(), input.clone()))
                    }
                    _ => None,
                })
                .collect();

            // res から必要な情報を取得後に、消費する
            history.push(res.into());

            // ツールの実行リクエストがないかつ、 struct output を求められている場合は、respond を強制的に利用させる
            if tool_calls.is_empty() && has_respond {
                tool_choice = ToolChoice::Specific("respond".into());
                continue;
            }

            // 実行リクエストがきたツールを全て実行
            let mut blocks: Vec<MessageBlock> = Vec::new();
            let mut tasks = Vec::new();
            let limit = self.max_tool_calls_per_turn as usize;

            // 残りをこのターンで動かすツールに等分する。
            // 各ツールが取り分を守る限り、木全体でも budget を超えない
            let running = tool_calls.len().min(limit).max(1) as u32;
            let share = budget
                .limit
                .saturating_sub(input_tokens.saturating_add(output_tokens))
                / running;

            for (i, (id, name, input)) in tool_calls.into_iter().enumerate() {
                // 上限を超えた分は実行しない
                // tool_use には必ず tool_result を返す必要があるので、拒否も結果として返し、
                // 次のターンで LLM に絞り込ませる
                if i >= limit {
                    blocks.push(MessageBlock::ToolResult {
                        tool_use_id: id,
                        content: Kind::TooManyToolCalls
                            .with(format!(
                                "at most {limit} tools may be called per turn; \
                                 `{name}` was not run. call fewer tools at a time."
                            ))
                            .to_string(),
                        is_error: true,
                    });
                    continue;
                }

                let tool = tool_map.get(&name).ok_or(Kind::ToolNotFound.default())?;
                tool.set_token_budget(share);

                // Future を直接配列にいれることで id, name, input, tool の借用をなくし（ move ）、ループの外で実行可能にする
                tasks.push(async move {
                    // ツールが固まっても実行を打ち切れるように、必ず制限時間を被せる
                    // 時間切れはツールのエラーと同様に LLM へ差し戻し、リトライや断念を委ねる
                    let limit = tool.timeout().unwrap_or(self.default_tool_timeout);

                    let block = match tokio::time::timeout(limit, tool.execute(input)).await {
                        Ok(Ok(v)) => MessageBlock::ToolResult {
                            tool_use_id: id,
                            content: v.to_string(),
                            is_error: false,
                        },
                        Ok(Err(e)) => MessageBlock::ToolResult {
                            tool_use_id: id,
                            content: e.to_string(),
                            is_error: true,
                        },
                        Err(_) => MessageBlock::ToolResult {
                            tool_use_id: id,
                            content: Kind::ToolTimeout
                                .with(format!("tool `{}` timed out after {:?}", name, limit))
                                .to_string(),
                            is_error: true,
                        },
                    };

                    // サブエージェント等、ツール自身が LLM を消費した分を回収して合算する
                    let (tool_input_tokens, tool_output_tokens) = tool.sub_agent_usage();

                    Ok::<_, AgentError>((block, tool_input_tokens, tool_output_tokens))
                });
            }

            let results = join_all(tasks)
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            for (block, tool_input_tokens, tool_output_tokens) in results {
                input_tokens += tool_input_tokens;
                output_tokens += tool_output_tokens;
                budget.record(tool_input_tokens, tool_output_tokens);
                blocks.push(block);
            }

            // ツール実行結果を履歴につめて再度 invoke に回す
            history.push(Message::user(blocks));
        }
    }

    pub async fn run(
        &self,
        model: &Model,
        input: Vec<Input>,
    ) -> Result<AgentResult<String>, AgentError> {
        self.run_within(model, input, u32::MAX, None).await
    }

    /// 親エージェントから渡された残り予算の範囲で走らせる。
    /// 自身の `max_total_tokens` と、渡された値の小さい方に従う
    pub(crate) async fn run_within(
        &self,
        model: &Model,
        input: Vec<Input>,
        budget: u32,
        sink: Option<(&AtomicU32, &AtomicU32)>,
    ) -> Result<AgentResult<String>, AgentError> {
        let tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        self.loop_call(
            model,
            input,
            &self.system_prompt,
            &tool_refs,
            Budget {
                limit: self.max_total_tokens.min(budget),
                sink,
            },
            |res| {
                // ToolUse でない場合は完了とする
                (!matches!(res.stop_reason, StopReason::ToolUse)).then(|| {
                    let text = res
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ResultBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    Ok(text)
                })
            },
        )
        .await
    }

    pub async fn run_typed<T>(
        &self,
        model: &Model,
        input: Vec<Input>,
    ) -> Result<AgentResult<T>, AgentError>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static,
    {
        let mut tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        let respond = tool::RespondTool::<T>::new();
        tool_refs.push(&respond);

        let system = format!(
            "{}\n\n最終的な回答は必ず respond ツールを呼び出して返してください。",
            self.system_prompt
        );

        self.loop_call(
            model,
            input,
            &system,
            &tool_refs,
            Budget {
                limit: self.max_total_tokens,
                sink: None,
            },
            |res| {
                // ToolUse で respond を指定している場合のみ完了とする
                // struct output では respond ツールの input スキーマを生成させ、それを最終出力に利用する
                res.content
                    .iter()
                    .find_map(|b| match b {
                        ResultBlock::ToolUse { name, input, .. } if name == "respond" => {
                            Some(input.clone())
                        }
                        _ => None,
                    })
                    .map(|input| serde_json::from_value(input).map_err(Into::into))
            },
        )
        .await
    }
}

pub struct AgentBuilder {
    system_prompt: String,
    max_tokens: u32,
    max_turns: u32,
    max_total_tokens: u32,
    max_tool_calls_per_turn: u32,
    default_tool_timeout: Duration,
    tools: Vec<Box<dyn tool::Tool>>,
    use_models: Vec<Model>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_tokens: 1024,
            max_turns: 10,
            max_total_tokens: 500_000,
            max_tool_calls_per_turn: 8,
            default_tool_timeout: Duration::from_secs(60),
            tools: Vec::new(),
            use_models: vec![Model::BedrockClaudeSonnet46],
        }
    }
}
impl AgentBuilder {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = turns;
        self
    }

    pub fn max_total_tokens(mut self, tokens: u32) -> Self {
        self.max_total_tokens = tokens;
        self
    }

    pub fn max_tool_calls_per_turn(mut self, calls: u32) -> Self {
        self.max_tool_calls_per_turn = calls;
        self
    }

    pub fn default_tool_timeout(mut self, timeout: Duration) -> Self {
        self.default_tool_timeout = timeout;
        self
    }

    pub fn add_tool(mut self, tool: Box<dyn tool::Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn use_models(mut self, models: Vec<Model>) -> Self {
        self.use_models = models;
        self
    }

    pub async fn build(self) -> Result<Agent, AgentError> {
        if self.use_models.is_empty() {
            return Err(ValidationException.with("at least one model must be specified"));
        }

        let mut providers: HashMap<Model, Box<dyn llm::LLM>> = HashMap::new();
        for model in self.use_models {
            match model {
                Model::BedrockClaudeSonnet46 => {
                    providers.insert(model, Box::new(bedrock::Adapter::new().await));
                }
            }
        }

        Ok(Agent {
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
            max_total_tokens: self.max_total_tokens,
            max_tool_calls_per_turn: self.max_tool_calls_per_turn,
            default_tool_timeout: self.default_tool_timeout,
            tools: self.tools,
            providers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::{InvokeResult, Usage};
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
            max_tokens: 1024,
            max_turns: 10,
            max_total_tokens: u32::MAX,
            max_tool_calls_per_turn: 8,
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

    struct TokenSpendingTool {
        usage: (u32, u32),
    }
    #[async_trait]
    impl tool::Tool for TokenSpendingTool {
        fn name(&self) -> String {
            "sub".into()
        }
        fn description(&self) -> String {
            "test".into()
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        fn sub_agent_usage(&self) -> (u32, u32) {
            self.usage
        }
        async fn execute(&self, _: Value) -> Result<Value, AgentError> {
            Ok(json!("done"))
        }
    }

    #[tokio::test]
    async fn max_turns_stops_an_llm_that_keeps_calling_tools() {
        let (mut agent, rec) = agent_with(vec![calls(&["noop"], (1, 1))], vec![instant("noop")]);
        agent.max_turns = 3;

        let err = agent.run(&MODEL, go()).await.unwrap_err();

        assert_eq!(err.kind, Kind::MaxTurnsExceeded);
        assert_eq!(rec.calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn token_budget_stops_the_loop_before_max_turns_does() {
        let (mut agent, rec) = agent_with(vec![calls(&["noop"], (100, 50))], vec![instant("noop")]);
        agent.max_turns = 100;
        agent.max_total_tokens = 300;

        let err = agent.run(&MODEL, go()).await.unwrap_err();

        assert_eq!(err.kind, Kind::TokenBudgetExceeded);
        assert_eq!(rec.calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn an_exhausted_budget_still_returns_an_answer_that_was_reached() {
        let (mut agent, _) = agent_with(vec![ends("答え", (1000, 1000))], vec![]);
        agent.max_total_tokens = 10;

        let res = agent.run(&MODEL, go()).await.unwrap();

        assert_eq!(res.content, "答え");
        assert_eq!(res.input_tokens + res.output_tokens, 2000);
    }

    #[tokio::test]
    async fn calling_an_unregistered_tool_is_an_error() {
        let (agent, _) = agent_with(vec![calls(&["nope"], (1, 1))], vec![]);

        assert_eq!(
            agent.run(&MODEL, go()).await.unwrap_err().kind,
            Kind::ToolNotFound
        );
    }

    #[tokio::test]
    async fn tokens_spent_inside_a_tool_are_added_to_the_total() {
        let (agent, _) = agent_with(
            vec![calls(&["sub"], (10, 5)), ends("done", (1, 2))],
            vec![Box::new(TokenSpendingTool { usage: (700, 300) })],
        );

        let res = agent.run(&MODEL, go()).await.unwrap();

        assert_eq!(res.input_tokens, 10 + 700 + 1);
        assert_eq!(res.output_tokens, 5 + 300 + 2);
    }

    struct BudgetProbe {
        seen: Arc<Mutex<Vec<u32>>>,
    }
    #[async_trait]
    impl tool::Tool for BudgetProbe {
        fn name(&self) -> String {
            "sub".into()
        }
        fn description(&self) -> String {
            "test".into()
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        fn set_token_budget(&self, remaining: u32) {
            self.seen.lock().unwrap().push(remaining);
        }
        async fn execute(&self, _: Value) -> Result<Value, AgentError> {
            Ok(json!("done"))
        }
    }

    #[tokio::test]
    async fn the_remaining_budget_is_split_across_the_tools_of_a_turn() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (mut agent, _) = agent_with(
            vec![
                calls(&["sub", "sub", "sub", "sub"], (100, 100)),
                ends("done", (1, 1)),
            ],
            vec![Box::new(BudgetProbe { seen: seen.clone() })],
        );
        agent.max_total_tokens = 1_000;
        agent.max_tool_calls_per_turn = 2;

        agent.run(&MODEL, go()).await.unwrap();

        // 1000 - 200 消費済み = 残り 800 を、実行する 2 本で等分
        assert_eq!(*seen.lock().unwrap(), [400, 400]);
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
        agent.max_turns = 1000;
        agent.max_total_tokens = u32::MAX;
        (agent, rec)
    }

    #[tokio::test]
    async fn a_sub_agent_cannot_outspend_the_parent_budget() {
        // 子は無制限に設定してある（max_turns 1000 / 予算 u32::MAX）
        let (child, child_rec) = child_agent(calls(&["noop"], (100, 100)), Duration::ZERO);

        let (mut parent, _) = agent_with(
            vec![calls(&["research"], (10, 10)), ends("done", (1, 1))],
            vec![Box::new(tool::AgentTool::new(
                "research", "test", MODEL, child,
            ))],
        );
        parent.max_total_tokens = 5_000;

        let err = parent.run(&MODEL, go()).await.unwrap_err();
        assert_eq!(err.kind, Kind::TokenBudgetExceeded);

        // 子は自分の上限ではなく、親から渡された取り分で止まっている
        let child_spent = child_rec.calls.load(Ordering::Relaxed) * 200;
        assert!(child_spent <= 5_000, "子が {child_spent} 使った");
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

        let res = parent.run(&MODEL, go()).await.unwrap();

        // 打ち切りまでに完了した 2 回分が計上されていること
        assert_eq!(res.input_tokens, 1 + 100_000 + 1, "{res:?}");
    }

    #[tokio::test]
    async fn tool_calls_beyond_the_per_turn_cap_are_not_run() {
        let (mut agent, rec) = agent_with(
            vec![
                calls(&["sub", "sub", "sub", "sub", "sub"], (1, 1)),
                ends("done", (1, 1)),
            ],
            vec![Box::new(TokenSpendingTool { usage: (100, 100) })],
        );
        agent.max_tool_calls_per_turn = 2;

        let res = agent.run(&MODEL, go()).await.unwrap();

        assert_eq!(res.input_tokens, 1 + 100 + 100 + 1);
        assert_eq!(res.output_tokens, 1 + 100 + 100 + 1);

        let history = rec.last_history.lock().unwrap();
        let blocks = &history.last().unwrap().content;
        assert_eq!(blocks.len(), 5);
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
        agent.run(&MODEL, go()).await.unwrap();
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

        let res = agent.run(&MODEL, go()).await.unwrap();
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
        agent.run(&MODEL, go()).await.unwrap();

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

        let res = agent.run_typed::<Answer>(&MODEL, go()).await.unwrap();

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
                .run_typed::<Answer>(&MODEL, go())
                .await
                .unwrap_err()
                .kind,
            Kind::UnknownException
        );
    }
}
