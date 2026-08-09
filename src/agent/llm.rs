use crate::agent::tool::Tool;
use crate::types::agent as types;
use crate::types::agent::ToolChoice;
use crate::types::errors::AgentError;
use crate::types::model::Model;
use async_trait::async_trait;

#[async_trait]
#[allow(clippy::upper_case_acronyms)]
pub trait LLM: Send + Sync {
    async fn invoke(
        &self,
        model: &Model,
        system_prompt: &str,
        max_tokens: u32,
        messages: &[types::Message],
        tools: &[&dyn Tool],
        tool_choice: &ToolChoice,
    ) -> Result<types::InvokeResult, AgentError>;
}
