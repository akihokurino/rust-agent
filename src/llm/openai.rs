use crate::agent;
use crate::agent::llm::LLM;
use crate::types::ReasoningEffort;
use crate::types::agent::{
    InvokeResult, Message, MessageBlock, ResultBlock, StopReason, ToolChoice, Usage,
};
use crate::types::errors::{AgentError, Kind};
use crate::types::model::Model;
use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct Adapter {
    client: reqwest::Client,
}

impl Adapter {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for Adapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLM for Adapter {
    async fn invoke(
        &self,
        model: &Model,
        system_prompt: &str,
        max_tokens: u32,
        messages: &[Message],
        tools: &[&dyn agent::tool::Tool],
        tool_choice: &ToolChoice,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Result<InvokeResult, AgentError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Kind::ValidationException.with("OPENAI_API_KEY is not set"))?;
        let mut body = json!({
            "model": model.to_string(),
            "instructions": system_prompt,
            "max_output_tokens": max_tokens,
            "input": input_items(messages),
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(
                tools
                    .iter()
                    .map(|t| {
                        json!({
                            "type": "function",
                            "name": t.name(),
                            "description": t.description(),
                            "parameters": t.input_schema(),
                            "strict": false,
                        })
                    })
                    .collect(),
            );
        }
        if let ToolChoice::Specific(name) = tool_choice {
            body["tool_choice"] = json!({ "type": "function", "name": name });
        }
        if let Some(effort) = reasoning_effort {
            body["reasoning"] = json!({ "effort": effort.as_str() });
        }

        let response = self
            .client
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            let kind = match status.as_u16() {
                400 => Kind::ValidationException,
                403 => Kind::ModelAccessDeniedException,
                _ => Kind::UnknownException,
            };
            return Err(kind.with(format!("OpenAI HTTP {status}: {detail}")));
        }
        let response: Value = response
            .json()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        from_response(&response)
    }
}

fn input_items(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();
    for message in messages {
        let role = match message.role {
            crate::types::agent::Role::User => "user",
            crate::types::agent::Role::Assistant => "assistant",
        };
        let mut content = Vec::new();
        let mut tail = Vec::new();
        for block in &message.content {
            match block {
                MessageBlock::Text { text } => content.push(json!({ "type": "input_text", "text": text })),
                MessageBlock::Pdf { data } => content.push(json!({
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": format!("data:application/pdf;base64,{}", base64::engine::general_purpose::STANDARD.encode(data)),
                })),
                MessageBlock::ToolUse { id, name, input } => tail.push(json!({
                    "type": "function_call", "call_id": id, "name": name, "arguments": input.to_string(),
                })),
                MessageBlock::ToolResult { tool_use_id, content, .. } => tail.push(json!({
                    "type": "function_call_output", "call_id": tool_use_id, "output": content,
                })),
                MessageBlock::Thinking { .. } | MessageBlock::RedactedThinking { .. } => {}
            }
        }
        if !content.is_empty() {
            items.push(json!({ "role": role, "content": content }));
        }
        items.extend(tail);
    }
    items
}

fn from_response(response: &Value) -> Result<InvokeResult, AgentError> {
    let output = response["output"]
        .as_array()
        .ok_or_else(|| Kind::UnknownException.with("OpenAI response has no output array"))?;
    let mut content = Vec::new();
    let mut has_tool_call = false;
    for item in output {
        match item["type"].as_str() {
            Some("message") => {
                if let Some(parts) = item["content"].as_array() {
                    for part in parts {
                        if part["type"] == "output_text"
                            && let Some(text) = part["text"].as_str()
                        {
                            content.push(ResultBlock::Text {
                                text: text.to_owned(),
                            });
                        }
                    }
                }
            }
            Some("function_call") => {
                let id = item["call_id"].as_str().ok_or_else(|| {
                    Kind::UnknownException.with("OpenAI function call has no call_id")
                })?;
                let name = item["name"].as_str().ok_or_else(|| {
                    Kind::UnknownException.with("OpenAI function call has no name")
                })?;
                let arguments = item["arguments"].as_str().ok_or_else(|| {
                    Kind::UnknownException.with("OpenAI function call has no arguments")
                })?;
                content.push(ResultBlock::ToolUse {
                    id: id.to_owned(),
                    name: name.to_owned(),
                    input: serde_json::from_str(arguments)?,
                });
                has_tool_call = true;
            }
            _ => {}
        }
    }
    let stop_reason = if has_tool_call {
        StopReason::ToolUse
    } else if response["incomplete_details"]["reason"] == "max_output_tokens" {
        StopReason::MaxTokens
    } else {
        StopReason::EndTurn
    };
    let tokens = |name: &str| -> u32 {
        response["usage"][name]
            .as_u64()
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32
    };
    Ok(InvokeResult {
        content,
        stop_reason,
        usage: Usage {
            input_tokens: tokens("input_tokens"),
            output_tokens: tokens("output_tokens"),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::agent::Role;

    #[test]
    fn maps_messages_and_response() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![
                    MessageBlock::Text {
                        text: "hello".into(),
                    },
                    MessageBlock::ToolUse {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        input: json!({"x": 1}),
                    },
                ],
            },
            Message::user(vec![MessageBlock::ToolResult {
                tool_use_id: "call_1".into(),
                content: "ok".into(),
                is_error: false,
            }]),
        ];
        let items = input_items(&messages);
        assert_eq!(items[0]["content"][0]["text"], "hello");
        assert_eq!(items[1]["arguments"], "{\"x\":1}");
        assert_eq!(items[2]["call_id"], "call_1");

        let result = from_response(&json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                {"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{\"x\":2}"}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4}
        })).unwrap();
        assert!(matches!(result.stop_reason, StopReason::ToolUse));
        assert_eq!(result.content.len(), 2);
        assert_eq!(result.usage.input_tokens, 10);
    }
}
