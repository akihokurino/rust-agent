use crate::agent::types;
use base64::Engine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<MessageBlock>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
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

impl From<types::Message> for Message {
    fn from(m: types::Message) -> Self {
        Message {
            role: m.role.into(),
            content: m.content.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<types::Role> for Role {
    fn from(r: types::Role) -> Self {
        match r {
            types::Role::User => Role::User,
            types::Role::Assistant => Role::Assistant,
        }
    }
}

impl From<types::MessageBlock> for MessageBlock {
    fn from(b: types::MessageBlock) -> Self {
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
                is_error,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn to_json(b: types::MessageBlock) -> serde_json::Value {
        serde_json::to_value(MessageBlock::from(b)).unwrap()
    }

    #[test]
    fn roles_are_lowercase() {
        let m: Message = types::Message::user(vec![]).into();
        assert_eq!(serde_json::to_value(&m).unwrap()["role"], "user");

        let m = Message {
            role: types::Role::Assistant.into(),
            content: vec![],
        };
        assert_eq!(serde_json::to_value(&m).unwrap()["role"], "assistant");
    }

    #[test]
    fn text_block() {
        assert_eq!(
            to_json(types::MessageBlock::Text {
                text: "こんにちは".into()
            }),
            json!({ "type": "text", "text": "こんにちは" })
        );
    }

    #[test]
    fn tool_use_block() {
        assert_eq!(
            to_json(types::MessageBlock::ToolUse {
                id: "tu_1".into(),
                name: "fetch_url".into(),
                input: json!({ "url": "https://example.com" }),
            }),
            json!({
                "type": "tool_use",
                "id": "tu_1",
                "name": "fetch_url",
                "input": { "url": "https://example.com" },
            })
        );
    }

    #[test]
    fn tool_result_block_carries_the_error_flag() {
        assert_eq!(
            to_json(types::MessageBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "timed out".into(),
                is_error: true,
            }),
            json!({
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "timed out",
                "is_error": true,
            })
        );
    }

    #[test]
    fn pdf_becomes_a_base64_document() {
        assert_eq!(
            to_json(types::MessageBlock::Pdf {
                data: b"%PDF-1.4".to_vec()
            }),
            json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQ=",
                },
            })
        );
    }

    #[test]
    fn response_is_parsed_into_internal_types() {
        let raw = json!({
            "content": [
                { "type": "text", "text": "調べます" },
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "web_search",
                    "input": { "query": "サイボウズ 会社概要" },
                },
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1234, "output_tokens": 56 },
        });

        let parsed: InvokeResponse = serde_json::from_value(raw).unwrap();
        let res: types::InvokeResult = parsed.into();

        assert!(matches!(res.stop_reason, types::StopReason::ToolUse));
        assert_eq!(res.usage.input_tokens, 1234);
        assert_eq!(res.usage.output_tokens, 56);
        assert_eq!(res.content.len(), 2);
        match &res.content[1] {
            types::ResultBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "web_search");
                assert_eq!(input["query"], "サイボウズ 会社概要");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn every_stop_reason_is_understood() {
        for (raw, expected) in [
            ("end_turn", types::StopReason::EndTurn),
            ("tool_use", types::StopReason::ToolUse),
            ("max_tokens", types::StopReason::MaxTokens),
            ("stop_sequence", types::StopReason::StopSequence),
        ] {
            let parsed: StopReason = serde_json::from_value(json!(raw)).unwrap();
            let got: types::StopReason = parsed.into();
            assert_eq!(
                std::mem::discriminant(&got),
                std::mem::discriminant(&expected),
                "{raw}"
            );
        }
    }
}
