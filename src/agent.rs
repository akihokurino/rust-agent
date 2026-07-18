use crate::llm::bedrock;
use crate::types::errors::AgentError;
use crate::types::errors::Kind::ValidationException;
use crate::types::model::Model;
use std::collections::HashMap;

pub mod llm;
pub mod tool;

pub struct Agent {
    pub system_prompt: String,
    pub max_tokens: u32,
    pub tools: Vec<Box<dyn tool::Tool>>,
    providers: HashMap<Model, Box<dyn llm::LLM>>,
}
impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    pub async fn run(
        &self,
        model: &Model,
        messages: Vec<llm::types::Message>,
    ) -> Result<String, AgentError> {
        panic!("not implemented");
    }
}

pub struct AgentBuilder {
    system_prompt: String,
    max_tokens: u32,
    tools: Vec<Box<dyn tool::Tool>>,
    use_models: Vec<Model>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_tokens: 1024,
            tools: Vec::new(),
            use_models: vec![Model::BedrockClaudeSonnet46],
        }
    }
}
impl AgentBuilder {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn add_tool(mut self, tool: Box<dyn tool::Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn use_models(mut self, models: Vec<Model>) -> Self {
        self.use_models = models;
        self
    }

    pub async fn build(self) -> Result<Agent, AgentError> {
        if self.use_models.is_empty() {
            return Err(ValidationException
                .with("at least one model must be specified")
                .into());
        }

        let mut providers: HashMap<Model, Box<dyn llm::LLM>> = HashMap::new();
        for model in self.use_models {
            match model {
                Model::BedrockClaudeSonnet46 => {
                    providers.insert(model, Box::new(bedrock::Adapter::new().await));
                }
            }
        }

        Ok(Agent {
            system_prompt: self.system_prompt,
            max_tokens: self.max_tokens,
            tools: self.tools,
            providers,
        })
    }
}
