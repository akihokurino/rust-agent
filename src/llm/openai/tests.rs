use super::*;
use crate::types::agent::Role;

#[test]
fn maps_messages_and_response() {
    let messages = vec![
        Message {
            role: Role::Assistant,
            content: vec![
                MessageBlock::Text {
                    text: "hello".into(),
                },
                MessageBlock::ToolUse {
                    id: "call_1".into(),
                    name: "lookup".into(),
                    input: json!({"x": 1}),
                },
            ],
        },
        Message::user(vec![MessageBlock::ToolResult {
            tool_use_id: "call_1".into(),
            content: "ok".into(),
            is_error: false,
        }]),
    ];
    let items = input_items(&messages);
    assert_eq!(items[0]["content"][0]["text"], "hello");
    assert_eq!(items[1]["arguments"], "{\"x\":1}");
    assert_eq!(items[2]["call_id"], "call_1");

    let result = from_response(&json!({
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
            {"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{\"x\":2}"}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4}
    })).unwrap();
    assert!(matches!(result.stop_reason, StopReason::ToolUse));
    assert_eq!(result.content.len(), 2);
    assert_eq!(result.usage.input_tokens, 10);
}

#[test]
fn forwards_pdf_as_a_base64_input_file() {
    let messages = [Message::user(vec![
        MessageBlock::Text {
            text: "Read this".into(),
        },
        MessageBlock::Pdf {
            data: b"PDF".to_vec(),
        },
    ])];

    let items = input_items(&messages);

    assert_eq!(
        items[0],
        json!({
            "role": "user",
            "content": [
                { "type": "input_text", "text": "Read this" },
                {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": "data:application/pdf;base64,UERG",
                }
            ],
        })
    );
}

#[test]
fn reports_the_output_limit_and_keeps_usage() {
    let result = from_response(&json!({
        "output": [
            { "type": "message", "content": [{ "type": "output_text", "text": "partial" }] }
        ],
        "incomplete_details": { "reason": "max_output_tokens" },
        "usage": { "input_tokens": 7, "output_tokens": 12 },
    }))
    .unwrap();

    assert!(matches!(result.stop_reason, StopReason::MaxTokens));
    assert_eq!(
        (result.usage.input_tokens, result.usage.output_tokens),
        (7, 12)
    );
}

#[test]
fn rejects_missing_output_and_invalid_function_arguments() {
    for response in [
        json!({ "output": null }),
        json!({
            "output": [
                { "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{" }
            ]
        }),
    ] {
        let error = from_response(&response).unwrap_err();

        assert_eq!(error.kind, Kind::UnknownException);
    }
}
