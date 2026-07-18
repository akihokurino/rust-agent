use async_trait::async_trait;
use rust_agent::{Agent, AgentError, Input, Kind, Model, Tool};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

const MODEL: Model = Model::BedrockClaudeSonnet46;

// ---------- tools ----------

fn strip_html(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(50_000)
        .collect()
}

#[derive(Deserialize, JsonSchema)]
struct FetchUrlInput {
    /// 取得するページのURL
    url: String,
}

struct FetchUrl;

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
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: FetchUrlInput = serde_json::from_value(input)?;
        let resp = reqwest::Client::new()
            .get(&args.url)
            .header("User-Agent", "Mozilla/5.0 (compatible; CompanyResearcherBot/1.0)")
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        let status = resp.status().as_u16();
        let html = resp.text().await.map_err(Kind::UnknownException.from_srcf())?;
        Ok(json!({ "content": strip_html(&html), "status": status }))
    }
}

#[derive(Deserialize, JsonSchema)]
struct WebSearchInput {
    /// 検索クエリ
    query: String,
}

struct WebSearch;

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
    async fn execute(&self, input: Value) -> Result<Value, AgentError> {
        let args: WebSearchInput = serde_json::from_value(input)?;
        let api_key = std::env::var("SERPER_API_KEY").unwrap_or_default();
        let resp = reqwest::Client::new()
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .json(&json!({ "q": args.query, "gl": "jp", "hl": "ja", "num": 5 }))
            .send()
            .await
            .map_err(Kind::UnknownException.from_srcf())?
            .json::<Value>()
            .await
            .map_err(Kind::UnknownException.from_srcf())?;
        Ok(json!({ "results": resp["organic"] }))
    }
}

// ---------- schemas ----------

#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
struct CompanyInfo {
    /// 会社名（正式名称）
    name: String,
    /// 本社所在地の都道府県。不明なら空文字。
    prefecture: String,
    /// 最も代表的な業種を1つ。不明なら空文字。
    industry: String,
    /// 本社住所（都道府県含む完全な表記）。不明なら空文字。
    address: String,
    /// 代表者名。不明なら空文字。
    ceo_name: String,
    /// 設立年月日。YYYY-MM-DD 形式。不明なら空文字。
    established_date: String,
    /// 資本金（公式表記のまま）。不明なら空文字。
    capital: String,
    /// 従業員数（公式表記のまま）。不明なら空文字。
    employee_num: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
enum EmploymentType {
    FullTime,
    Contract,
    PartTime,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
struct CareerItem {
    /// 入社年月（YYYY-MM）
    start_ym: String,
    /// 退社年月（YYYY-MM）。現職なら空文字。
    end_ym: String,
    /// 企業名
    company_name: String,
    /// 業種。不明なら空文字。
    industry: String,
    /// 職種。不明なら "Other"。
    occupation: String,
    /// 雇用形態
    employment_type: EmploymentType,
    /// 役職。明示が無ければ「メンバー」など妥当な値。
    post: String,
    /// 経歴詳細（業務内容・実績の要約）。255文字以内。
    detail: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
struct CareerExtraction {
    /// 時系列で古い順に並べた経歴の配列
    careers: Vec<CareerItem>,
}

// ---------- main ----------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // 1. ネット情報の抽出（会社URL → fetch_url / web_search → 構造化）
    let company_agent = Agent::builder()
        .system_prompt(
            "あなたは会社の公式サイトURLから会社情報を抽出するエージェントです。\n\
             fetch_url でページ本文を取得し、必要なら web_search で会社概要ページを探します。\n\
             不明な項目は空文字。推測で埋めないこと。",
        )
        .max_tokens(4096)
        .add_tool(Box::new(FetchUrl))
        .add_tool(Box::new(WebSearch))
        .build()
        .await?;

    let company = company_agent
        .run_typed::<CompanyInfo>(
            &MODEL,
            vec![Input::Text(
                "次の会社URLから会社情報を抽出してください: https://www.cybozu.co.jp/".into(),
            )],
        )
        .await?;
    println!("[company] {:#?}", company);

    // 2. PDFの抽出（職務経歴書PDF → 構造化）
    let career_agent = Agent::builder()
        .system_prompt(
            "あなたは添付された職務経歴書PDFから career 情報を抽出するエージェントです。\n\
             careers は時系列で古い順。detail は255文字以内。全項目を必ず返すこと。",
        )
        .max_tokens(4096)
        .build()
        .await?;

    let pdf = std::fs::read("examples/sample_career.pdf")?;
    let career = career_agent
        .run_typed::<CareerExtraction>(
            &MODEL,
            vec![
                Input::Pdf(pdf),
                Input::Text("添付の職務経歴書PDFから career 情報を抽出してください。".into()),
            ],
        )
        .await?;
    println!("[career] {:#?}", career);

    Ok(())
}
