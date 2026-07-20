# rust_agent

A small agent library for Rust: an LLM tool-calling loop with structured output, built on Amazon Bedrock.

The scope is deliberately narrow. The library provides the loop, a tool abstraction, structured output, and limits that
bound how far the loop can run — nothing else. There is no streaming, no session storage, no provider abstraction.

## Requirements

- Rust 1.97+ (edition 2024)
- AWS credentials with `bedrock:InvokeModel` permission

The Bedrock client is constructed internally and cannot be swapped from outside. Region is `ap-northeast-1`, and the
only model is `Model::BedrockClaudeSonnet46` (`jp.anthropic.claude-sonnet-4-6`).

## Limits

| Builder option            | Default   | What it bounds                                     |
|---------------------------|-----------|----------------------------------------------------|
| `max_tokens`              | 1024      | Output tokens of a single model call               |
| `max_turns`               | 10        | Iterations of the tool-calling loop                |
| `max_total_tokens`        | 500,000   | Tokens accumulated across one `run`                |
| `max_tool_calls_per_turn` | 8         | Tools executed in a single turn                    |
| `default_tool_timeout`    | 60s       | Wall-clock time of one tool execution              |

`max_total_tokens` is checked at the top of each turn, against what has been spent so far. The consequences are worth
being explicit about:

- The first model call of a `run` is always made. A budget smaller than that call does not prevent it.
- The ceiling is therefore `max_total_tokens` plus one turn, not `max_total_tokens` exactly.
- A `run` that finishes in a single turn is not bounded by it at all.

When an agent is used as a tool, the remaining budget is divided equally among the tools executed in that turn, and each
sub-agent runs under the smaller of its own `max_total_tokens` and the share it was given. Spend is recorded per turn, so
a sub-agent that is cut short still reports what it used. This makes the limit hold across the whole tree, not just the
root.

### What is not bounded

Input size. `max_tokens` caps output only, and nothing inspects an `Input` before it is sent — a large `Input::Pdf`
becomes a large, billable request on the very first call. The only ceiling is the model's context window, beyond which
Bedrock rejects the request.

This is deliberate: what counts as a reasonable input depends on the domain, which the library does not know. Callers
that accept untrusted input should bound it themselves — by byte size before decoding, or by page count — rather than
relying on `max_total_tokens` to do it.