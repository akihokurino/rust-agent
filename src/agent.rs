use crate::agent::types::{Message, MessageBlock, ResultBlock, StopReason, ToolChoice};
use crate::llm::bedrock;
use crate::types::agent::AgentResult;
use crate::types::errors::AgentError;
use crate::types::errors::Kind::ValidationException;
use crate::types::model::Model;
use crate::Kind;
use std::collections::HashMap;

pub mod llm;
pub mod tool;
pub mod types;

pub struct Agent {
    pub system_prompt: String,
    pub max_tokens: u32,
    pub max_turns: u32,
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
        message: &str,
        system: &str,
        tool_refs: &[&dyn tool::Tool],
        finish: impl Fn(&types::InvokeResult) -> Option<Result<O, AgentError>>,
    ) -> Result<AgentResult<O>, AgentError> {
        let llm = self.providers.get(model).ok_or(Kind::ModelNotConfigured)?;
        let tool_map = tool_refs
            .iter()
            .map(|t| (t.name(), t))
            .collect::<HashMap<_, _>>();

        let mut history = vec![Message::user_text(message)];
        let mut turns: u32 = 0;
        let mut input_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;
        let mut tool_choice = ToolChoice::Auto;
        let has_respond = tool_map.contains_key("respond");

        loop {
            if turns >= self.max_turns {
                return Err(Kind::MaxTurnsExceeded.default());
            }
            turns += 1;

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

            if let Some(result) = finish(&res) {
                let content = result?;
                return Ok(AgentResult {
                    content,
                    input_tokens,
                    output_tokens,
                });
            }

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

            history.push(res.into());

            if tool_calls.is_empty() && has_respond {
                tool_choice = ToolChoice::Tool("respond".into());
                continue;
            }

            let mut results = Vec::new();
            for (id, name, input) in tool_calls {
                let tool = tool_map.get(&name).ok_or(Kind::ToolNotFound.default())?;
                let block = match tool.execute(input).await {
                    Ok(v) => MessageBlock::ToolResult {
                        tool_use_id: id,
                        content: v.to_string(),
                        is_error: false,
                    },
                    Err(e) => MessageBlock::ToolResult {
                        tool_use_id: id,
                        content: e.to_string(),
                        is_error: true,
                    },
                };
                results.push(block);
            }
            history.push(Message::user_tool_results(results));

            tool_choice = ToolChoice::Auto;
        }
    }

    pub async fn run(
        &self,
        model: &Model,
        message: &str,
    ) -> Result<AgentResult<String>, AgentError> {
        let tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();

        self.loop_call(model, message, &self.system_prompt, &tool_refs, |res| {
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
        message: &str,
    ) -> Result<AgentResult<T>, AgentError>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static,
    {
        let respond = tool::RespondTool::<T>::new();
        let mut tool_refs: Vec<&dyn tool::Tool> = self.tools.iter().map(|t| t.as_ref()).collect();
        tool_refs.push(&respond);

        let system = format!(
            "{}\n\n最終的な回答は必ず respond ツールを呼び出して返してください。",
            self.system_prompt
        );

        self.loop_call(model, message, &system, &tool_refs, |res| {
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
    tools: Vec<Box<dyn tool::Tool>>,
    use_models: Vec<Model>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_tokens: 1024,
            max_turns: 10,
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
            return Err(ValidationException
                .with("at least one model must be specified")
                .into());
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
            tools: self.tools,
            providers,
        })
    }
}
