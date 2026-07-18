mod types;

use crate::agent;
use crate::agent::llm::LLM;
use crate::agent::types::ToolChoice;
use crate::llm::bedrock::types::InvokeResponse;
use crate::types::errors::{AgentError, Kind};
use crate::types::model::Model;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use serde_json::{json, Value};

const REGION: &str = "ap-northeast-1";

#[derive(Clone, Debug)]
pub struct Adapter {
    client: Client,
}

impl Adapter {
    pub async fn new() -> Self {
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(REGION)
            .load()
            .await;
        let client = Client::new(&config);
        Self { client }
    }
}

#[async_trait]
impl LLM for Adapter {
    async fn invoke(
        &self,
        model: &Model,
        system_prompt: &str,
        max_tokens: u32,
        messages: &[agent::types::Message],
        tools: &[&dyn agent::tool::Tool],
        tool_choice: &ToolChoice,
    ) -> Result<agent::types::InvokeResult, AgentError> {
        let messages: Vec<types::Message> = messages.iter().map(|m| m.clone().into()).collect();
        let mut body = json!({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        if !tools.is_empty() {
            let specs: Vec<Value> = tools.iter().map(|t| t.spec()).collect();
            body["tools"] = json!(specs);
        }
        match tool_choice {
            ToolChoice::Auto => (),
            ToolChoice::Specific(name) => {
                body["tool_choice"] = json!({ "type": "tool", "name": name })
            }
        }

        let resp = self
            .client
            .invoke_model()
            .model_id(model.to_string())
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(serde_json::to_vec(&body)?))
            .send()
            .await?;

        let out: InvokeResponse = serde_json::from_slice(resp.body().as_ref())?;
        Ok(out.into())
    }
}

impl From<SdkError<InvokeModelError, HttpResponse>> for AgentError {
    #[track_caller]
    fn from(value: SdkError<InvokeModelError, HttpResponse>) -> Self {
        use InvokeModelError::*;
        match value {
            SdkError::ServiceError(se) => match se.into_err() {
                AccessDeniedException(inner) => Kind::ModelAccessDeniedException.from_src(inner),
                ValidationException(inner) => Kind::ValidationException.from_src(inner),
                other => Kind::UnknownException.from_src(other),
            },
            other => Kind::UnknownException.from_src(other),
        }
    }
}
