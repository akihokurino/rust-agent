use crate::agent::tool::Tool;
use crate::agent::types;
use crate::types::errors::AgentError;
use crate::types::model::Model;
use async_trait::async_trait;

#[async_trait]
pub trait LLM: Send + Sync {
    async fn invoke(
        &self,
        model: &Model,
        system_prompt: &str,
        max_tokens: u32,
        messages: &[types::Message],
        tools: &[Box<dyn Tool>],
    ) -> Result<types::InvokeResult, AgentError>;
}
