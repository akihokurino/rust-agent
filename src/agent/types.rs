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
