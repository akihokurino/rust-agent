use crate::types::errors::AgentError;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::Value;
use std::marker::PhantomData;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn input_schema(&self) -> Value;
    async fn execute(&self, input: Value) -> Result<Value, AgentError>;

    fn spec(&self) -> Value {
        serde_json::json!({
            "name": self.name(),
            "description": self.description(),
            "input_schema": self.input_schema(),
        })
    }
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
