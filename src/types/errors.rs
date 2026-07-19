use derive_more::Display;
use std::fmt::{Display, Formatter};
use std::panic::Location;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Display)]
pub enum Kind {
    ModelAccessDeniedException,
    ValidationException,
    ModelNotConfigured,
    ToolNotFound,
    ToolTimeout,
    MaxTurnsExceeded,
    TokenBudgetExceeded,
    UnknownException,
}
impl Kind {
    #[track_caller]
    pub fn default(self) -> AgentError {
        let msg = match self {
            Kind::ModelAccessDeniedException => "access to the requested Bedrock model was denied",
            Kind::ValidationException => "the request sent to Bedrock was invalid",
            Kind::ModelNotConfigured => "the requested model is not configured on this agent",
            Kind::ToolNotFound => "the requested tool was not found",
            Kind::ToolTimeout => "the tool did not finish within the allotted time",
            Kind::MaxTurnsExceeded => "the maximum number of turns has been exceeded",
            Kind::TokenBudgetExceeded => "the token budget for this run has been exhausted",
            Kind::UnknownException => "an unexpected error occurred",
        };

        AgentError {
            kind: self,
            msg: Some(msg.to_string()),
            src: None,
            location: Location::caller(),
        }
    }

    #[track_caller]
    pub fn with(self, msg: impl Into<String>) -> AgentError {
        AgentError {
            kind: self,
            msg: Some(msg.into()),
            src: None,
            location: Location::caller(),
        }
    }

    #[inline]
    #[track_caller]
    pub fn withf<T>(self) -> impl FnOnce(T) -> AgentError
    where
        T: Into<String>,
    {
        let location = Location::caller();
        move |v| AgentError {
            kind: self,
            msg: Some(v.into()),
            src: None,
            location,
        }
    }

    #[track_caller]
    pub fn from_src(self, src: impl std::error::Error + Send + Sync + 'static) -> AgentError {
        AgentError {
            kind: self,
            msg: None,
            src: Some(Arc::from(src)),
            location: Location::caller(),
        }
    }

    #[inline]
    #[track_caller]
    pub fn from_srcf<T>(self) -> impl FnOnce(T) -> AgentError
    where
        T: std::error::Error + Send + Sync + 'static,
    {
        let location = Location::caller();
        move |v| AgentError {
            kind: self,
            msg: None,
            src: Some(Arc::from(v)),
            location,
        }
    }

    #[track_caller]
    pub fn into_err(self) -> AgentError {
        self.into()
    }
}

#[derive(Debug, Clone)]
pub struct AgentError {
    pub kind: Kind,
    pub msg: Option<String>,
    pub src: Option<Arc<dyn std::error::Error + Send + Sync>>,
    pub location: &'static Location<'static>,
}
impl AgentError {
    #[track_caller]
    pub fn with_src(self, src: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            src: Some(Arc::from(src)),
            location: Location::caller(),
            ..self
        }
    }

    #[track_caller]
    pub fn with_box_src(self, src: Box<dyn std::error::Error + Send + Sync + 'static>) -> Self {
        Self {
            src: Some(Arc::from(src)),
            location: Location::caller(),
            ..self
        }
    }
}
impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}{}",
            self.kind,
            self.msg
                .as_ref()
                .map(|v| format!(": {}", v))
                .unwrap_or_default(),
            self.src
                .as_ref()
                .map(|v| format!(": {}", v))
                .unwrap_or_default()
        )
    }
}
impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.src
            .as_ref()
            .map(|v| v.as_ref() as &(dyn std::error::Error + 'static))
    }
}
impl From<Kind> for AgentError {
    #[track_caller]
    fn from(kind: Kind) -> Self {
        Self {
            kind,
            msg: None,
            src: None,
            location: Location::caller(),
        }
    }
}
impl From<String> for AgentError {
    #[track_caller]
    fn from(value: String) -> Self {
        Self {
            kind: Kind::UnknownException,
            msg: Some(value),
            src: None,
            location: Location::caller(),
        }
    }
}

macro_rules! impl_from_err_to_unknown_err {
    ($T:ty) => {
        impl From<$T> for crate::types::errors::AgentError {
            fn from(v: $T) -> Self {
                crate::types::errors::Kind::UnknownException.from_src(v)
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use impl_from_err_to_unknown_err;

impl_from_err_to_unknown_err!(serde_json::Error);
