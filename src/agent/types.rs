use crate::Input;

#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<MessageBlock>,
}
impl Message {
    pub fn user(content: Vec<MessageBlock>) -> Self {
        Message {
            role: Role::User,
            content,
        }
    }
}
impl From<InvokeResult> for Message {
    fn from(value: InvokeResult) -> Self {
        Message {
            role: Role::Assistant,
            content: value.content.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone)]
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
    Pdf {
        data: Vec<u8>,
    },
}
impl From<ResultBlock> for MessageBlock {
    fn from(value: ResultBlock) -> Self {
        match value {
            ResultBlock::Text { text } => MessageBlock::Text { text },
            ResultBlock::ToolUse { id, name, input } => MessageBlock::ToolUse { id, name, input },
        }
    }
}
impl From<Input> for MessageBlock {
    fn from(i: Input) -> Self {
        match i {
            Input::Text(text) => MessageBlock::Text { text },
            Input::Pdf(data) => MessageBlock::Pdf { data },
        }
    }
}

#[derive(Debug, Clone)]
pub struct InvokeResult {
    pub content: Vec<ResultBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub enum ResultBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone)]
pub enum ToolChoice {
    Auto,
    Specific(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_messages_take_the_user_role() {
        let m = Message::user(vec![MessageBlock::Text { text: "hi".into() }]);
        assert!(matches!(m.role, Role::User));
        assert_eq!(m.content.len(), 1);
    }

    #[test]
    fn input_text_becomes_a_text_block() {
        let b: MessageBlock = Input::Text("こんにちは".into()).into();
        match b {
            MessageBlock::Text { text } => assert_eq!(text, "こんにちは"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn input_pdf_keeps_the_bytes_intact() {
        let b: MessageBlock = Input::Pdf(vec![0x25, 0x50, 0x44, 0x46]).into();
        match b {
            MessageBlock::Pdf { data } => assert_eq!(data, b"%PDF"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn invoke_results_are_folded_back_as_assistant_messages() {
        let res = InvokeResult {
            content: vec![
                ResultBlock::Text {
                    text: "考え中".into(),
                },
                ResultBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "fetch_url".into(),
                    input: serde_json::json!({ "url": "https://example.com" }),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
            },
        };

        let m: Message = res.into();

        // 履歴に積み直すので、必ず assistant 側になる
        assert!(matches!(m.role, Role::Assistant));
        assert_eq!(m.content.len(), 2);
        match &m.content[1] {
            MessageBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "fetch_url");
                assert_eq!(input["url"], "https://example.com");
            }
            other => panic!("{other:?}"),
        }
    }
}
