# rust_agent

A small agent library for Rust: an LLM tool-calling loop with structured output, supporting Amazon Bedrock and OpenAI.

The scope is deliberately narrow. The library provides the loop, a tool abstraction, structured output, and limits that
bound how far the loop can run — nothing else. There is no streaming or session storage.

## Requirements

- Rust 1.97+ (edition 2024)
- AWS credentials with `bedrock:InvokeModel` permission
- `OPENAI_API_KEY` when using an OpenAI model

Install the development tools and Git hook from the repository root.

```sh
mise install
mise exec -- lefthook install
mise exec -- make check
```

The Bedrock client is constructed internally and cannot be swapped from outside. Region is `ap-northeast-1`, and the
Bedrock models are `Model::BedrockClaudeSonnet46` (`jp.anthropic.claude-sonnet-4-6`) and
`Model::BedrockClaudeSonnet55` (`global.anthropic.claude-sonnet-5-5`).
Bedrock thinking blocks are kept in the returned history for subsequent turns. When reusing that history,
keep the system prompt, tool definitions, and earlier messages unchanged.

OpenAI models are `Model::Gpt56Terra` and `Model::Gpt56Luna`. Select them with
`Agent::builder().use_models(vec![Model::Gpt56Luna])`; optionally set
`.reasoning_effort(ReasoningEffort::High)`.

OpenAI-only clients can omit the AWS SDK with `default-features = false, features = ["builtin-tools"]`.
