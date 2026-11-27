use serde_json::Value;
use std::future::Future;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::time::timeout;

const TEST_ENDPOINT: &str = "RUST_AGENT_TEST_BEDROCK_ENDPOINT";
const ACCESS_KEY: &str = "local-test";

pub(super) enum Response {
    Ok(Value),
    AccessDenied,
}

pub(super) async fn exercise<const N: usize>(
    name: &str,
    replies: [Response; N],
    scenario: impl Future<Output = ()>,
) -> Option<Vec<Value>> {
    if let Ok(endpoint) = std::env::var(TEST_ENDPOINT) {
        let url = reqwest::Url::parse(&endpoint).unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        assert!(url.port().is_some());
        assert_eq!(std::env::var("AWS_ENDPOINT_URL").unwrap(), endpoint);
        assert_eq!(
            std::env::var("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").unwrap(),
            endpoint
        );
        assert_eq!(
            std::env::var("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS").unwrap(),
            "false"
        );
        assert_eq!(std::env::var("AWS_ACCESS_KEY_ID").unwrap(), ACCESS_KEY);
        scenario.await;
        return None;
    }

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    // The SDK reads its configuration from the child environment. Each test gets
    // a fresh process with no inherited AWS profiles, credentials, or proxies.
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg(name)
        .env_clear()
        .env(TEST_ENDPOINT, &endpoint)
        .env("AWS_ENDPOINT_URL", &endpoint)
        .env("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", &endpoint)
        .env("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "false")
        .env("AWS_ACCESS_KEY_ID", ACCESS_KEY)
        .env("AWS_SECRET_ACCESS_KEY", "local-test-secret")
        .env("AWS_CONFIG_FILE", "/dev/null")
        .env("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .env("AWS_MAX_ATTEMPTS", "1")
        .kill_on_drop(true)
        .output();

    let (_, requests) = timeout(Duration::from_secs(20), async {
        tokio::try_join!(
            async {
                let output = child.await.map_err(|error| error.to_string())?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                if output.status.success() && stdout.contains("1 passed") {
                    Ok(())
                } else {
                    Err(format!(
                        "child did not pass {name}:\n{stdout}\n{}",
                        String::from_utf8_lossy(&output.stderr)
                    ))
                }
            },
            async { Ok(respond(listener, replies).await) }
        )
    })
    .await
    .expect("Bedrock integration scenario timed out")
    .unwrap_or_else(|error| panic!("{error}"));
    Some(requests)
}

async fn respond(listener: TcpListener, replies: impl IntoIterator<Item = Response>) -> Vec<Value> {
    let mut received = Vec::new();
    for reply in replies {
        let (mut connection, _) = listener.accept().await.unwrap();
        received.push(read_request(&mut connection).await);
        let (status, headers, body) = match reply {
            Response::Ok(body) => ("200 OK", "", body.to_string()),
            Response::AccessDenied => (
                "403 Forbidden",
                "x-amzn-errortype: AccessDeniedException\r\n",
                r#"{"message":"fixture denied access"}"#.to_owned(),
            ),
        };
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
            body.len()
        );
        connection.write_all(response.as_bytes()).await.unwrap();
        connection.shutdown().await.unwrap();
    }
    received
}

async fn read_request(connection: &mut TcpStream) -> Value {
    const REQUEST_LIMIT: usize = 64 * 1024;
    let mut request = Vec::new();
    let mut chunk = [0; 1024];
    let headers_end = loop {
        let count = connection.read(&mut chunk).await.unwrap();
        assert!(count > 0, "connection ended before the headers");
        request.extend_from_slice(&chunk[..count]);
        assert!(
            request.len() <= REQUEST_LIMIT,
            "request exceeds fixture limit"
        );
        if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = std::str::from_utf8(&request[..headers_end]).unwrap();
    assert!(headers.starts_with("POST /model/jp.anthropic.claude-sonnet-4-6/invoke "));
    let length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>().unwrap())
        .expect("Bedrock request has no content length");
    assert!(
        length <= REQUEST_LIMIT - headers_end,
        "request exceeds fixture limit"
    );
    let end = headers_end + length;
    if request.len() < end {
        let start = request.len();
        request.resize(end, 0);
        connection.read_exact(&mut request[start..]).await.unwrap();
    }
    serde_json::from_slice(&request[headers_end..end]).unwrap()
}
