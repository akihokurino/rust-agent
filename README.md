# rust_agent

A small agent library for Rust: an LLM tool-calling loop with structured output, built on Amazon Bedrock.

The scope is deliberately narrow. The library provides the loop, a tool abstraction, structured output, and hard limits
on cost — nothing else. There is no streaming, no session storage, no provider abstraction.

## Requirements

- Rust 1.97+ (edition 2024)
- AWS credentials with `bedrock:InvokeModel` permission

The Bedrock client is constructed internally and cannot be swapped from outside. Region is `ap-northeast-1`, and the
only model is `Model::BedrockClaudeSonnet46` (`jp.anthropic.claude-sonnet-4-6`).