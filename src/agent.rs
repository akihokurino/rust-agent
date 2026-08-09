use crate::llm::bedrock;
use crate::types::agent::AgentResult;
use crate::types::agent::{
    InvokeResult, Message, MessageBlock, ResultBlock, StopReason, ToolChoice,
};
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
#[cfg(test)]
mod tests;
pub mod tool;

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
        prev_history: Vec<Message>,
        system: &str,
        tool_refs: &[&dyn tool::Tool],
        finish: impl Fn(&InvokeResult) -> Option<Result<(O, bool), AgentError>>,
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
        let mut history = prev_history
            .into_iter()
            .chain([Message::user(content)])
            .collect::<Vec<_>>();
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
                // 最終ターンの場合はここで終了するので、最後の回答を history につめる
                // ただし、構造化出力の場合は最終回答は respond ツールへのリクエストの形なので、この場合はhistoryに含めない
                let (result, is_include_last_message) = result?;
                if is_include_last_message {
                    history.push(res.into());
                }

                let (input, output) = budget.usage();
                return Ok(AgentResult {
                    content: result,
                    history,
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
        history: Option<Vec<Message>>,
    ) -> Result<AgentResult<String>, AgentError> {
        let tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        self.with_budget(self.loop_call(
            model,
            input,
            history.unwrap_or_default(),
            &self.system_prompt,
            &tool_refs,
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
                    Ok((text, true))
                })
            },
        ))
        .await
    }

    pub async fn run_typed<T>(
        &self,
        model: &Model,
        input: Vec<Input>,
        history: Option<Vec<Message>>,
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

        self.with_budget(self.loop_call(
            model,
            input,
            history.unwrap_or_default(),
            &system,
            &tool_refs,
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
                    .map(|input| {
                        serde_json::from_value(input)
                            .map(|v| (v, false))
                            .map_err(Into::into)
                    })
            },
        ))
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
