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
    pub fn user_text(text: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: vec![MessageBlock::Text { text: text.into() }],
        }
    }
    pub fn user_tool_results(results: Vec<MessageBlock>) -> Self {
        Message {
            role: Role::User,
            content: results,
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
}
impl From<ResultBlock> for MessageBlock {
    fn from(value: ResultBlock) -> Self {
        match value {
            ResultBlock::Text { text } => MessageBlock::Text { text },
            ResultBlock::ToolUse { id, name, input } => MessageBlock::ToolUse { id, name, input },
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

pub enum ToolChoice {
    Auto,
    Tool(String),
}
