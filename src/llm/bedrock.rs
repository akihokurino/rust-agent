pub mod types;

use crate::llm::bedrock::types::InvokeResponse;
use crate::types::errors::{AgentError, Kind};
use crate::types::model::Model;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use serde_json::json;

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

    pub async fn invoke(
        &self,
        model: &Model,
        message: &str,
        max_tokens: u32,
    ) -> Result<InvokeResponse, AgentError> {
        let body = json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                { "role": "user", "content": message }
            ]
        });

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
        Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    // 実際にAWSを叩くので、通常の cargo test では走らせない（#[ignore]）
    #[tokio::test]
    #[ignore = "hits real Bedrock; run with --ignored"]
    async fn smoke_invoke() -> anyhow::Result<()> {
        let adapter = Adapter::new().await;
        let out = adapter
            .invoke(
                &Model::BedrockClaudeSonnet46,
                "こんにちは、一言で返して",
                1024,
            )
            .await?;

        println!("response: {:?}", out);
        Ok(())
    }
}
