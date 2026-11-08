use derive_more::Display;

#[derive(Debug, Eq, PartialEq, Display, Hash)]
pub enum Model {
    #[display("jp.anthropic.claude-sonnet-4-6")]
    BedrockClaudeSonnet46,
    #[display("gpt-5.6-terra")]
    Gpt56Terra,
    #[display("gpt-5.6-luna")]
    Gpt56Luna,
}
