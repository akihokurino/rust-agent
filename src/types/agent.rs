#[derive(Debug, Clone)]
pub struct AgentResult<T> {
    pub content: T,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub enum Input {
    Text(String),
    Pdf(Vec<u8>),
}
