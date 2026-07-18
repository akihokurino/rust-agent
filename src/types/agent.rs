#[derive(Debug, Clone)]
pub struct AgentResult {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}
