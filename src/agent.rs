use crate::agent::types::{Message, MessageBlock, ResultBlock, StopReason, ToolChoice};
use crate::llm::bedrock;
use crate::types::agent::AgentResult;
use crate::types::errors::AgentError;
use crate::types::errors::Kind::ValidationException;
use crate::types::model::Model;
use crate::{Input, Kind};
use futures::future::join_all;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

pub mod llm;
pub mod tool;
pub mod types;

// 1 回の run が使ってよいトークン上限と、全体の input, output 消費量
// ルートで 1 つ作り、サブエージェントも同じものを加算・参照する
struct Budget {
    limit: u32,
    input: AtomicU32,
    output: AtomicU32,
}
impl Budget {
    fn record(&self, input: u32, output: u32) {
        self.input.fetch_add(input, Ordering::Relaxed);
        self.output.fetch_add(output, Ordering::Relaxed);
    }

    fn usage(&self) -> (u32, u32) {
        (
            self.input.load(Ordering::Relaxed),
            self.output.load(Ordering::Relaxed),
        )
    }

    fn valid(&self) -> Result<(), AgentError> {
        let (i, o) = self.usage();
        let spent = i.saturating_add(o);
        if spent >= self.limit {
            Err(Kind::TokenBudgetExceeded
                .with(format!("consumed {spent} tokens (budget: {})", self.limit)))
        } else {
            Ok(())
        }
    }
}
// 1つのグリーンスレッド単位で Budget を管理する
tokio::task_local! {
    static BUDGET: Arc<Budget>;
}

const MAX_TOOL_CALLS_PER_TURN: u32 = 10;
const MAX_TURNS: u32 = 10;
const MAX_TOKENS: u32 = 1024;
const MAX_TOTAL_TOKENS: u32 = 500_000;

pub struct Agent {
    pub system_prompt: String,
    pub max_turns: u32,
    pub max_tokens: u32,
    pub max_total_tokens: u32,
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
        finish: impl Fn(&types::InvokeResult) -> Option<Result<O, AgentError>>,
    ) -> Result<AgentResult<O>, AgentError> {
        // 指定モデルから利用する LLM アダプターを決定
        let llm = self.providers.get(model).ok_or(Kind::ModelNotConfigured)?;

        let tool_map = tool_refs
            .iter()
            .map(|t| (t.name(), t))
            .collect::<HashMap<_, _>>();

        // ルートで作られた予算。サブエージェントとして呼ばれた場合は親のものが見える
        let budget = BUDGET
            .try_with(Arc::clone)
            .map_err(|_| Kind::TokenBudgetExceeded.with("no budget in scope"))?;
        let started = budget.usage();

        let content: Vec<MessageBlock> = input.into_iter().map(Into::into).collect();
        let mut history = vec![Message::user(content)];
        let mut turns: u32 = 0;
        let mut tool_choice = ToolChoice::Auto;

        // struct output を利用している場合は true となる
        let has_respond = tool_map.contains_key("respond");

        loop {
            // 異常時のために最大試行回数を決めておく
            if turns >= self.max_turns {
                return Err(Kind::MaxTurnsExceeded.default());
            }

            // ターン数だけでは1ターンあたりのツール呼び出し数で消費が積み上がるため、トークン量でも上限を設ける
            budget.valid()?;

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

            budget.record(res.usage.input_tokens, res.usage.output_tokens);

            // 完了条件を満たすか検証する
            // 満たしていた場合はそこで結果を返す
            if let Some(result) = finish(&res) {
                let (input, output) = budget.usage();
                return Ok(AgentResult {
                    content: result?,
                    input_tokens: input - started.0,
                    output_tokens: output - started.1,
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
            let limit = MAX_TOOL_CALLS_PER_TURN as usize;

            for (i, (id, name, input)) in tool_calls.into_iter().enumerate() {
                // 上限を超えた分は実行しない
                // tool_use には必ず tool_result を返す必要があるので、拒否も結果として返し、次のターンで LLM に絞り込ませる
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

                    Ok::<_, AgentError>(block)
                });
            }

            let tasks = join_all(tasks)
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            blocks.extend(tasks);

            // ツール実行結果を履歴につめて再度 invoke に回す
            history.push(Message::user(blocks));
        }
    }

    async fn with_budget<F, O>(&self, f: F) -> Result<AgentResult<O>, AgentError>
    where
        F: Future<Output = Result<AgentResult<O>, AgentError>>,
    {
        if BUDGET.try_with(|_| ()).is_ok() {
            return f.await;
        }
        BUDGET
            .scope(
                Arc::new(Budget {
                    limit: self.max_total_tokens,
                    input: AtomicU32::new(0),
                    output: AtomicU32::new(0),
                }),
                f,
            )
            .await
    }

    pub async fn run(
        &self,
        model: &Model,
        input: Vec<Input>,
    ) -> Result<AgentResult<String>, AgentError> {
        let tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        self.with_budget(
            self.loop_call(model, input, &self.system_prompt, &tool_refs, |res| {
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
            }),
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

        self.with_budget(self.loop_call(model, input, &system, &tool_refs, |res| {
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
        }))
        .await
    }
}

pub struct AgentBuilder {
    system_prompt: String,
    max_turns: u32,
    max_tokens: u32,
    max_total_tokens: u32,
    default_tool_timeout: Duration,
    tools: Vec<Box<dyn tool::Tool>>,
    use_models: Vec<Model>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_turns: MAX_TURNS,
            max_tokens: MAX_TOKENS,
            max_total_tokens: MAX_TOTAL_TOKENS,
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

    /// 1 回の Agent 実行における最大ターン数
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = turns;
        self
    }

    /// 1 回のモデル呼び出しが返す出力トークンの上限
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// 1 回の run で積算してよいトークンの上限
    pub fn max_total_tokens(mut self, tokens: u32) -> Self {
        self.max_total_tokens = tokens;
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

    pub fn add_sub_agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        model: Model,
        sub_agent: SubAgent,
    ) -> Self {
        self.tools.push(Box::new(tool::AgentTool::new(
            name,
            description,
            model,
            sub_agent.0,
        )));
        self
    }

    pub fn add_nested_agent(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        model: Model,
        sub_agent: Agent,
    ) -> Self {
        self.tools.push(Box::new(tool::AgentTool::new(
            name,
            description,
            model,
            sub_agent,
        )));
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
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            max_total_tokens: self.max_total_tokens,
            default_tool_timeout: self.default_tool_timeout,
            tools: self.tools,
            providers,
        })
    }
}

pub struct SubAgent(Agent);

impl SubAgent {
    pub fn builder() -> SubAgentBuilder {
        SubAgentBuilder(AgentBuilder::default())
    }
}

pub struct SubAgentBuilder(AgentBuilder);

impl SubAgentBuilder {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.0 = self.0.system_prompt(prompt);
        self
    }

    /// 1 回のモデル呼び出しが返す出力トークンの上限
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.0 = self.0.max_tokens(tokens);
        self
    }

    pub fn default_tool_timeout(mut self, timeout: Duration) -> Self {
        self.0 = self.0.default_tool_timeout(timeout);
        self
    }

    pub fn add_tool(mut self, tool: Box<dyn tool::Tool>) -> Self {
        self.0 = self.0.add_tool(tool);
        self
    }

    pub fn use_models(mut self, models: Vec<Model>) -> Self {
        self.0 = self.0.use_models(models);
        self
    }

    pub async fn build(self) -> Result<SubAgent, AgentError> {
        Ok(SubAgent(self.0.build().await?))
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

        let err = agent.run(&MODEL, go()).await.unwrap_err();

        assert_eq!(err.kind, Kind::MaxTurnsExceeded);
        assert_eq!(rec.calls.load(Ordering::Relaxed), MAX_TURNS);
    }

    #[tokio::test]
    async fn token_budget_stops_the_loop_before_max_turns_does() {
        let (mut agent, rec) = agent_with(vec![calls(&["noop"], (100, 50))], vec![instant("noop")]);
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
    async fn a_sub_agents_tokens_are_included_in_the_parent_result() {
        let (child, _) = child_agent(ends("child done", (700, 300)), Duration::ZERO);

        let (parent, _) = agent_with(
            vec![calls(&["research"], (10, 5)), ends("done", (1, 2))],
            vec![Box::new(tool::AgentTool::new(
                "research", "test", MODEL, child,
            ))],
        );

        let res = parent.run(&MODEL, go()).await.unwrap();

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

        let err = parent.run(&MODEL, go()).await.unwrap_err();
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

        let res = parent.run(&MODEL, go()).await.unwrap();

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

        agent.run(&MODEL, go()).await.unwrap();

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
