#[cfg(test)]
mod tests;

use crate::types::agent as types;
use base64::Engine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct Message<'a> {
    pub role: Role,
    pub content: Vec<MessageBlock<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageBlock<'a> {
    Text {
        text: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a serde_json::Value,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: &'a str,
        is_error: bool,
    },
    Document {
        source: DocumentSource,
    },
}

#[derive(Debug, Serialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: &'static str,
    data: String,
}

impl<'a> From<&'a types::Message> for Message<'a> {
    fn from(m: &'a types::Message) -> Self {
        Message {
            role: (&m.role).into(),
            content: m.content.iter().map(Into::into).collect(),
        }
    }
}

impl From<&types::Role> for Role {
    fn from(r: &types::Role) -> Self {
        match r {
            types::Role::User => Role::User,
            types::Role::Assistant => Role::Assistant,
        }
    }
}

impl<'a> From<&'a types::MessageBlock> for MessageBlock<'a> {
    fn from(b: &'a types::MessageBlock) -> Self {
        match b {
            types::MessageBlock::Text { text } => MessageBlock::Text { text },
            types::MessageBlock::ToolUse { id, name, input } => {
                MessageBlock::ToolUse { id, name, input }
            }
            types::MessageBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => MessageBlock::ToolResult {
                tool_use_id,
                content,
                is_error: *is_error,
            },
            types::MessageBlock::Pdf { data } => MessageBlock::Document {
                source: DocumentSource {
                    source_type: "base64",
                    media_type: "application/pdf",
                    data: base64::engine::general_purpose::STANDARD.encode(data),
                },
            },
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct InvokeResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl From<InvokeResponse> for types::InvokeResult {
    fn from(r: InvokeResponse) -> Self {
        types::InvokeResult {
            content: r.content.into_iter().map(Into::into).collect(),
            stop_reason: r.stop_reason.into(),
            usage: r.usage.into(),
        }
    }
}

impl From<ContentBlock> for types::ResultBlock {
    fn from(b: ContentBlock) -> Self {
        match b {
            ContentBlock::Text { text } => types::ResultBlock::Text { text },
            ContentBlock::ToolUse { id, name, input } => {
                types::ResultBlock::ToolUse { id, name, input }
            }
        }
    }
}

impl From<StopReason> for types::StopReason {
    fn from(s: StopReason) -> Self {
        match s {
            StopReason::EndTurn => types::StopReason::EndTurn,
            StopReason::ToolUse => types::StopReason::ToolUse,
            StopReason::MaxTokens => types::StopReason::MaxTokens,
            StopReason::StopSequence => types::StopReason::StopSequence,
        }
    }
}

impl From<Usage> for types::Usage {
    fn from(u: Usage) -> Self {
        types::Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
        }
    }
}
