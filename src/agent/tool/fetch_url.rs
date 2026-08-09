#[cfg(test)]
mod tests;

use crate::agent::tool::Tool;
use crate::types::errors::{AgentError, Kind};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(serde::Deserialize, JsonSchema)]
struct FetchUrlInput {
    url: String,
}

pub struct FetchUrl;
const MAX_REDIRECTS: usize = 5;
#[async_trait]
impl Tool for FetchUrl {
    fn name(&self) -> &str {
        "fetch_url"
    }
    fn description(&self) -> &str {
        "指定されたURLのWebページ本文を取得します。企業の公式サイト等の情報収集に使います。"
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
        .trim_end_matches(']');
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| Kind::ValidationException.with("url has no port"))?;

    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host, port))
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
        .resolve(host, addr)
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(Kind::UnknownException.from_srcf())?;

    Ok((client, parsed))
}
/// インターネット上のアドレスとして到達を許すか。
/// `IpAddr::is_global` が unstable なため自前で判定している。
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
