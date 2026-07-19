use rust_agent::{AgentError, Kind};

#[test]
fn default_uses_the_kind_specific_message() {
    let e = Kind::MaxTurnsExceeded.default();
    assert_eq!(e.kind, Kind::MaxTurnsExceeded);
    assert_eq!(
        e.to_string(),
        "MaxTurnsExceeded: the maximum number of turns has been exceeded"
    );
}

#[test]
fn every_kind_has_a_default_message() {
    for kind in [
        Kind::ModelAccessDeniedException,
        Kind::ValidationException,
        Kind::ModelNotConfigured,
        Kind::ToolNotFound,
        Kind::ToolTimeout,
        Kind::MaxTurnsExceeded,
        Kind::TokenBudgetExceeded,
        Kind::UnknownException,
    ] {
        let e = kind.default();
        assert!(e.msg.is_some(), "{kind}");
        assert!(e.to_string().starts_with(&format!("{kind}: ")));
    }
}

#[test]
fn with_replaces_the_message() {
    let e = Kind::ToolTimeout.with("tool `x` timed out after 3s");
    assert_eq!(e.to_string(), "ToolTimeout: tool `x` timed out after 3s");
}

#[test]
fn display_appends_the_source() {
    let e = Kind::UnknownException.from_src(std::io::Error::other("boom"));
    assert!(e.msg.is_none());
    assert_eq!(e.to_string(), "UnknownException: boom");
}

#[test]
fn source_is_exposed_via_std_error() {
    use std::error::Error;
    let e = Kind::UnknownException.from_src(std::io::Error::other("boom"));
    assert_eq!(e.source().unwrap().to_string(), "boom");
}

#[test]
fn a_kind_with_no_message_displays_without_a_colon() {
    let e: AgentError = Kind::ToolNotFound.into();
    assert_eq!(e.to_string(), "ToolNotFound");
}

#[test]
fn location_points_at_the_call_site() {
    let expected = line!() + 1;
    let e = Kind::ToolNotFound.default();
    assert_eq!(e.location.line(), expected);
    assert!(e.location.file().ends_with("errors.rs"));
}

#[test]
fn serde_errors_become_unknown_exception() {
    let err = serde_json::from_str::<i32>("not json").unwrap_err();
    let e: AgentError = err.into();
    assert_eq!(e.kind, Kind::UnknownException);
}
