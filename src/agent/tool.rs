use crate::Input;
use crate::agent::Agent;
use crate::types::errors::{AgentError, Kind};
use crate::types::model::Model;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::{Value, json};
use std::marker::PhantomData;
use std::time::Duration;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn input_schema(&self) -> Value;
    fn timeout(&self) -> Option<Duration> {
        None
    }
    fn spec(&self) -> Value {
        json!({
            "name": self.name(),
            "description": self.description(),
            "input_schema": self.input_schema(),
        })
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError>;
}

pub(crate) struct RespondTool<T> {
    _phantom: PhantomData<T>,
}
impl<T> RespondTool<T> {
    pub(crate) fn new() -> Self {
        RespondTool {
            _phantom: PhantomData,
        }
    }
}
#[async_trait]
impl<T: JsonSchema + Send + Sync> Tool for RespondTool<T> {
    fn name(&self) -> String {
        "respond".into()
    }
    fn description(&self) -> String {
        "最終的な構造化された回答を返す".into()
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(T)).unwrap()
    }
    async fn execute(&self, _: Value) -> Result<Value, AgentError> {
        unreachable!()
    }
}

pub(crate) struct AgentTool {
    name: String,
    description: String,
    model: Model,
    sub_agent: Agent,
}
impl AgentTool {
    pub(crate) fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        model: Model,
        agent: Agent,
    ) -> Self {
        AgentTool {
            name: name.into(),
            description: description.into(),
            model,
            sub_agent: agent,
        }
    }
}
#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "サブエージェントに実行させたい指示",
                }
            },
            "required": ["prompt"],
        })
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let prompt = input
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| Kind::ValidationException.with("prompt is required"))?;

        let res = self
            .sub_agent
            .run(&self.model, vec![Input::Text(prompt.to_string())])
            .await?;

        Ok(json!(res.content))
    }
}

#[cfg(feature = "builtin-tools")]
#[derive(serde::Deserialize, JsonSchema)]
struct WebSearchInput {
    query: String,
}
#[cfg(feature = "builtin-tools")]
pub struct WebSearch {
    pub serper_api_key: Option<String>,
}
#[cfg(feature = "builtin-tools")]
#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> String {
        "web_search".into()
    }
    fn description(&self) -> String {
        "Google検索を行い、検索結果を取得します。".into()
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(WebSearchInput)).unwrap()
    }
    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(30))
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: WebSearchInput = serde_json::from_value(input)?;
        let api_key = self
            .serper_api_key
            .as_deref()
            .ok_or_else(|| Kind::ValidationException.with("serper_api_key is required"))?;
        let resp = reqwest::Client::new()
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .json(&json!({ "q": args.query, "gl": "jp", "hl": "ja", "num": 5 }))
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?
            .error_for_status()
            .map_err(Kind::UnknownException.from_srcf())?
            .json::<Value>()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        Ok(json!({ "results": resp["organic"] }))
    }
}

#[cfg(feature = "builtin-tools")]
#[derive(serde::Deserialize, JsonSchema)]
struct FetchUrlInput {
    url: String,
}

#[cfg(feature = "builtin-tools")]
pub struct FetchUrl;
#[cfg(feature = "builtin-tools")]
const MAX_REDIRECTS: usize = 5;
#[cfg(feature = "builtin-tools")]
#[async_trait]
impl Tool for FetchUrl {
    fn name(&self) -> String {
        "fetch_url".into()
    }
    fn description(&self) -> String {
        "指定されたURLのWebページ本文を取得します。企業の公式サイト等の情報収集に使います。".into()
    }
    fn input_schema(&self) -> Value {
        serde_json::to_value(schemars::schema_for!(FetchUrlInput)).unwrap()
    }
    fn timeout(&self) -> Option<Duration> {
        Some(Duration::from_secs(30))
    }
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: FetchUrlInput = serde_json::from_value(input)?;
        let mut url = args.url;

        for _ in 0..MAX_REDIRECTS {
            let (client, parsed) = guarded_client(&url).await?;

            let resp = client
                .get(parsed.clone())
                .header(
                    "User-Agent",
                    "Mozilla/5.0 (compatible; CompanyResearcherBot/1.0)",
                )
                .send()
                .await
                .map_err(Kind::UnknownException.from_srcf())?;

            let status = resp.status();
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);

            // Location 付きの 3xx のときだけ次のホップへ進み、再度検証をかける
            if let (true, Some(next)) = (status.is_redirection(), location) {
                url = parsed
                    .join(&next) // 相対 Location に対応
                    .map_err(Kind::ValidationException.from_srcf())?
                    .to_string();
                continue;
            }

            let html = resp
                .text()
                .await
                .map_err(Kind::UnknownException.from_srcf())?;

            return Ok(json!({
                "content": strip_html(&html),
                "status": status.as_u16(),
                "url": url, // リダイレクトの結果、実際に取得した URL
            }));
        }

        Err(Kind::ValidationException.with(format!("too many redirects (> {MAX_REDIRECTS})")))
    }
}
/// 宛先を検証した上で、その IP に固定した Client を組み立てる
#[cfg(feature = "builtin-tools")]
async fn guarded_client(url: &str) -> Result<(reqwest::Client, reqwest::Url), AgentError> {
    let parsed = reqwest::Url::parse(url).map_err(Kind::ValidationException.from_srcf())?;

    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(
            Kind::ValidationException.with(format!("unsupported url scheme: {}", parsed.scheme()))
        );
    }

    // IPv6 リテラルは host_str() が "[::1]" と括弧付きで返るため、
    // そのままでは名前解決に失敗して is_global の判定まで到達しない
    let host = parsed
        .host_str()
        .ok_or_else(|| Kind::ValidationException.with("url has no host"))?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| Kind::ValidationException.with("url has no port"))?;

    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(Kind::ValidationException.from_srcf())?
        .collect();

    // 1 つでも内部アドレスに解決されるなら拒否する
    let addr = *addrs
        .first()
        .ok_or_else(|| Kind::ValidationException.with(format!("could not resolve host: {host}")))?;
    if let Some(bad) = addrs.iter().find(|a| !is_global(&a.ip())) {
        return Err(Kind::ValidationException.with(format!(
            "refusing to fetch a non-global address: {} ({host})",
            bad.ip()
        )));
    }

    let client = reqwest::Client::builder()
        .resolve(&host, addr)
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(Kind::UnknownException.from_srcf())?;

    Ok((client, parsed))
}
/// インターネット上のアドレスとして到達を許すか。
/// `IpAddr::is_global` が unstable なため自前で判定している。
#[cfg(feature = "builtin-tools")]
fn is_global(ip: &std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            !(v4.is_loopback()                          // 127.0.0.0/8
                || v4.is_private()                      // 10/8, 172.16/12, 192.168/16
                || v4.is_link_local()                   // 169.254.0.0/16（メタデータ）
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.is_multicast()
                || o[0] == 0                            // 0.0.0.0/8
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24
                || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15
                || o[0] >= 240) // 240.0.0.0/4
        }
        IpAddr::V6(v6) => {
            // ::ffff:169.254.169.254 のような形で回避されるのを防ぐ
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_global(&IpAddr::V4(v4));
            }
            let s = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00    // fc00::/7 ULA
                || (s[0] & 0xffc0) == 0xfe80) // fe80::/10 link-local
        }
    }
}
/// HTML から本文だけを取り出す
/// 厳密なパースはせず、LLM に渡すトークンを削ることを目的とする
#[cfg(feature = "builtin-tools")]
fn strip_html(html: &str) -> String {
    // 表示されないのにトークンだけ食う要素は中身ごと捨てる
    const SKIP_TAGS: [&str; 4] = ["script", "style", "noscript", "template"];

    // タグ名を大文字小文字を無視して比較するための小文字版。
    // ASCII のみ変換するのでバイト長は変わらず、html と添字を共有できる
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let len = html.len();

    let mut out = String::new();
    let mut i = 0;

    while i < len {
        // タグ以外はそのまま本文として拾う
        if bytes[i] != b'<' {
            let start = i;
            while i < len && bytes[i] != b'<' {
                i += 1;
            }
            out.push_str(&html[start..i]);
            continue;
        }

        if lower[i..].starts_with("<!--") {
            i = lower[i..].find("-->").map(|p| i + p + 3).unwrap_or(len);
            continue;
        }

        // 閉じない '<' は本文とみなさず、そこで打ち切る
        let Some(p) = lower[i..].find('>') else { break };

        let inner = &lower[i + 1..i + p];
        let closing = inner.starts_with('/');
        let name = inner
            .trim_start_matches('/')
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or("");
        let skip = !closing && SKIP_TAGS.contains(&name);

        i += p + 1;
        // タグは単語の区切りとして扱う。<td>A</td><td>B</td> が "AB" になるのを防ぐ
        out.push(' ');

        if skip {
            // 対応する閉じタグまで中身ごと捨てる（閉じタグ自体は次の周回で処理される）
            let close = format!("</{name}");
            i = lower[i..].find(&close).map(|q| i + q).unwrap_or(len);
        }
    }

    let text = out.split_whitespace().collect::<Vec<_>>().join(" ");
    decode_entities(&text).chars().take(50_000).collect()
}
/// 頻出する文字参照だけを戻す
#[cfg(feature = "builtin-tools")]
fn decode_entities(s: &str) -> String {
    const ENTITIES: [(&str, &str); 6] = [
        ("&nbsp;", " "),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&#39;", "'"),
        // 二重デコードを避けるため最後に置く
        ("&amp;", "&"),
    ];
    let mut out = s.to_string();
    for (from, to) in ENTITIES {
        out = out.replace(from, to);
    }
    out
}

#[cfg(all(test, feature = "builtin-tools"))]
mod tests {
    use super::strip_html;

    #[test]
    fn drops_script_and_style_bodies() {
        let html = r#"<html><head>
            <style>body { color: red; }</style>
            <SCRIPT>var a = 1; if (a < 2) { alert("x") }</SCRIPT>
            </head><body><p>本文</p></body></html>"#;
        assert_eq!(strip_html(html), "本文");
    }

    #[test]
    fn separates_adjacent_cells() {
        assert_eq!(strip_html("<td>A</td><td>B</td>"), "A B");
    }

    #[test]
    fn drops_comments_and_decodes_entities() {
        assert_eq!(
            strip_html("<!-- 消える --><p>Q&amp;A&nbsp;&lt;x&gt;</p>"),
            "Q&A <x>"
        );
    }

    #[test]
    fn pathological_markup_always_terminates() {
        let cases: Vec<String> = vec![
            "<script></script>".into(),
            "<script>".into(),
            "<script></script></script>".into(),
            "<script><script></script>".into(),
            "<style></style><style></style>".into(),
            "<<<<<<<<".into(),
            "<!--".into(),
            "<!---->".into(),
            "<>".into(),
            "</>".into(),
            "<script".into(),
            "</script>".into(),
            "<SCRIPT></ScRiPt>".into(),
            "あ<script>い</script>う".into(),
            "<script>".repeat(1000),
            "<".repeat(10000),
            format!("<script>{}</script>", "<".repeat(1000)),
        ];
        for c in cases {
            let out = strip_html(&c);
            assert!(out.len() <= 50_000);
        }
    }

    #[test]
    fn handles_unclosed_tag() {
        assert_eq!(strip_html("<p>本文<broken"), "本文");
    }
}

#[cfg(test)]
mod agent_tool_tests {
    use super::*;

    async fn agent_tool() -> AgentTool {
        let sub = Agent::builder().build().await.unwrap();
        AgentTool::new(
            "research",
            "詳しく調べる",
            Model::BedrockClaudeSonnet46,
            sub,
        )
    }

    #[tokio::test]
    async fn exposes_a_prompt_and_nothing_else() {
        let t = agent_tool().await;

        assert_eq!(t.name(), "research");
        assert_eq!(t.description(), "詳しく調べる");

        let schema = t.input_schema();
        assert_eq!(schema["required"], json!(["prompt"]));
        assert_eq!(schema["properties"].as_object().unwrap().keys().len(), 1);
    }

    #[tokio::test]
    async fn rejects_a_missing_or_non_string_prompt() {
        let t = agent_tool().await;

        assert_eq!(
            t.execute(json!({})).await.unwrap_err().kind,
            Kind::ValidationException
        );
        assert_eq!(
            t.execute(json!({ "prompt": 42 })).await.unwrap_err().kind,
            Kind::ValidationException
        );
    }
}
