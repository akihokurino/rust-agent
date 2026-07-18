use derive_more::Display;

#[derive(Debug, PartialEq, Display)]
pub enum Model {
    #[display("jp.anthropic.claude-sonnet-4-6")]
    BedrockClaudeSonnet46,
}
