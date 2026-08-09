use super::*;
use serde_json::json;

fn to_json(b: types::MessageBlock) -> serde_json::Value {
    serde_json::to_value(MessageBlock::from(&b)).unwrap()
}

#[test]
fn roles_are_lowercase() {
    let user = types::Message::user(vec![]);
    let m: Message = (&user).into();
    assert_eq!(serde_json::to_value(&m).unwrap()["role"], "user");

    let m = Message {
        role: (&types::Role::Assistant).into(),
        content: vec![],
    };
    assert_eq!(serde_json::to_value(&m).unwrap()["role"], "assistant");
}

#[test]
fn text_block() {
    assert_eq!(
        to_json(types::MessageBlock::Text {
            text: "こんにちは".into()
        }),
        json!({ "type": "text", "text": "こんにちは" })
    );
}

#[test]
fn tool_use_block() {
    assert_eq!(
        to_json(types::MessageBlock::ToolUse {
            id: "tu_1".into(),
            name: "fetch_url".into(),
            input: json!({ "url": "https://example.com" }),
        }),
        json!({
            "type": "tool_use",
            "id": "tu_1",
            "name": "fetch_url",
            "input": { "url": "https://example.com" },
        })
    );
}

#[test]
fn tool_result_block_carries_the_error_flag() {
    assert_eq!(
        to_json(types::MessageBlock::ToolResult {
            tool_use_id: "tu_1".into(),
            content: "timed out".into(),
            is_error: true,
        }),
        json!({
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "timed out",
            "is_error": true,
        })
    );
}

#[test]
fn pdf_becomes_a_base64_document() {
    assert_eq!(
        to_json(types::MessageBlock::Pdf {
            data: b"%PDF-1.4".to_vec()
        }),
        json!({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQ=",
            },
        })
    );
}

#[test]
fn response_is_parsed_into_internal_types() {
    let raw = json!({
        "content": [
            { "type": "text", "text": "調べます" },
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "web_search",
                "input": { "query": "サイボウズ 会社概要" },
            },
        ],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 1234, "output_tokens": 56 },
    });

    let parsed: InvokeResponse = serde_json::from_value(raw).unwrap();
    let res: types::InvokeResult = parsed.into();

    assert!(matches!(res.stop_reason, types::StopReason::ToolUse));
    assert_eq!(res.usage.input_tokens, 1234);
    assert_eq!(res.usage.output_tokens, 56);
    assert_eq!(res.content.len(), 2);
    match &res.content[1] {
        types::ResultBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "tu_1");
            assert_eq!(name, "web_search");
            assert_eq!(input["query"], "サイボウズ 会社概要");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn every_stop_reason_is_understood() {
    for (raw, expected) in [
        ("end_turn", types::StopReason::EndTurn),
        ("tool_use", types::StopReason::ToolUse),
        ("max_tokens", types::StopReason::MaxTokens),
        ("stop_sequence", types::StopReason::StopSequence),
    ] {
        let parsed: StopReason = serde_json::from_value(json!(raw)).unwrap();
        let got: types::StopReason = parsed.into();
        assert_eq!(
            std::mem::discriminant(&got),
            std::mem::discriminant(&expected),
            "{raw}"
        );
    }
}
