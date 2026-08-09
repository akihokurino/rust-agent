#[cfg(feature = "builtin-tools")]
pub mod fetch_url;
#[cfg(test)]
mod tests;
#[cfg(feature = "builtin-tools")]
pub mod web_search;

use crate::Input;
use crate::agent::Agent;
use crate::types::errors::{AgentError, Kind};
use crate::types::model::Model;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::{Value, json};
use std::marker::PhantomData;
use std::time::Duration;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn input_schema(&self) -> Value;
    fn timeout(&self) -> Option<Duration> {
        None
    }
    fn spec(&self) -> Value {
        json!({
            "name": self.name(),
            "description": self.description(),
            "input_schema": self.input_schema(),
        })
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError>;
}

pub(crate) struct RespondTool<T> {
    _phantom: PhantomData<T>,
}
impl<T> RespondTool<T> {
    pub(crate) fn new() -> Self {
        RespondTool {
            _phantom: PhantomData,
        }
    }
}
#[async_trait]
impl<T: JsonSchema + Send + Sync> Tool for RespondTool<T> {
    fn name(&self) -> String {
        "respond".into()
    }
    fn description(&self) -> String {
        "最終的な構造化された回答を返す".into()
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(T)).unwrap()
    }
    async fn execute(&self, _: Value) -> Result<Value, AgentError> {
        unreachable!()
    }
}

pub(crate) struct AgentTool {
    name: String,
    description: String,
    model: Model,
    sub_agent: Agent,
}
impl AgentTool {
    pub(crate) fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        model: Model,
        agent: Agent,
    ) -> Self {
        AgentTool {
            name: name.into(),
            description: description.into(),
            model,
            sub_agent: agent,
        }
    }
}
#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "サブエージェントに実行させたい指示",
                }
            },
            "required": ["prompt"],
        })
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let prompt = input
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| Kind::ValidationException.with("prompt is required"))?;

        let res = self
            .sub_agent
            .run(&self.model, vec![Input::Text(prompt.to_string())], None)
            .await?;

        Ok(json!(res.content))
    }
}
