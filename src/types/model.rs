use derive_more::Display;

#[derive(Debug, Eq, PartialEq, Display, Hash)]
pub enum Model {
    #[cfg(feature = "bedrock")]
    #[display("jp.anthropic.claude-sonnet-4-6")]
    BedrockClaudeSonnet46,
    #[cfg(feature = "bedrock")]
    #[display("global.anthropic.claude-sonnet-5-5")]
    BedrockClaudeSonnet55,
    #[display("gpt-5.6-terra")]
    Gpt56Terra,
    #[display("gpt-5.6-luna")]
    Gpt56Luna,
}

impl Model {
    pub(crate) fn supports_forced_tool_choice(&self) -> bool {
        match self {
            #[cfg(feature = "bedrock")]
            Self::BedrockClaudeSonnet55 => false,
            _ => true,
        }
    }
}
