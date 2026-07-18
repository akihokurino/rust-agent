use derive_more::Display;

#[derive(Debug, Eq, PartialEq, Display, Hash)]
pub enum Model {
    #[display("jp.anthropic.claude-sonnet-4-6")]
    BedrockClaudeSonnet46,
}
