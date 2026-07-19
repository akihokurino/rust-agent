use crate::agent::types::{Message, MessageBlock, ResultBlock, StopReason, ToolChoice};
use crate::llm::bedrock;
use crate::types::agent::AgentResult;
use crate::types::errors::AgentError;
use crate::types::errors::Kind::ValidationException;
use crate::types::model::Model;
use crate::{Input, Kind};
use futures::future::join_all;
use std::collections::HashMap;
use std::time::Duration;

pub mod llm;
pub mod tool;
pub mod types;

pub struct Agent {
    pub system_prompt: String,
    pub max_tokens: u32,
    pub max_turns: u32,
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

            for (id, name, input) in tool_calls {
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
        let tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

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
        })
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

        self.loop_call(model, input, &system, &tool_refs, |res| {
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
        })
        .await
    }
}

pub struct AgentBuilder {
    system_prompt: String,
    max_tokens: u32,
    max_turns: u32,
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
            default_tool_timeout: self.default_tool_timeout,
            tools: self.tools,
            providers,
        })
    }
}
