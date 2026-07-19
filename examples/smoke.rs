use rust_agent::{Agent, FetchUrl, Input, Model, WebSearch};
use schemars::JsonSchema;
use serde::Deserialize;

const MODEL: Model = Model::BedrockClaudeSonnet46;

#[allow(dead_code)]
#[derive(Debug, Deserialize, JsonSchema)]
struct CompanyInfo {
    /// 会社名（正式名称）
    name: String,
    /// 本社所在地の都道府県。不明な場合は空文字。
    prefecture: String,
    /// 最も代表的な業種を1つ。不明な場合は空文字。
    industry: String,
    /// 本社住所（都道府県含む完全な表記）。不明な場合は空文字。
    address: String,
    /// 代表者名。不明な場合は空文字。
    ceo_name: String,
    /// 設立年月日。必ず YYYY-MM-DD 形式（例: 1950-04-01）。
    /// 日が不明なら 01 で補完してよい。年月日いずれかが完全に不明な場合は空文字。
    established_date: String,
    /// 資本金（公式表記をそのまま、例: "141,300百万円"）。不明な場合は空文字。
    capital: String,
    /// 従業員数（公式表記をそのまま、例: "5,800人 (連結)"）。不明な場合は空文字。
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
    /// 業種。不明な場合は空文字。
    industry: String,
    /// 職種。不明な場合は "Other"。
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // 1. companyInfoAgent: 会社URL → Web収集 → 構造化
    let company_agent = Agent::builder()
        .system_prompt(
            "あなたは会社の公式サイトURLから会社情報を抽出するエージェントです。\n\
             \n\
             入力として会社のURLが与えられます。以下の手順で必要な情報を集めてください。\n\
             \n\
             1. fetch_url ツールで与えられたURLのページ本文を取得する。\n\
             2. 会社情報・会社概要・企業情報・About 等のページに会社概要が無い場合は、\n\
             トップページのリンクや web_search で「会社名 会社概要」のように検索して\n\
             該当ページを見つけ、fetch_url で取得する。\n\
             3. 取得した情報を構造化された JSON で返す。\n\
             \n\
             注意:\n\
             - 情報が見つからない場合は空文字 \"\" を入れる。推測で値を埋めないこと。\n\
             - 全項目を必ず返すこと。",
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

    // 2. careerExtractorAgent: 職務経歴書PDF → 構造化（ツールなし）
    let career_agent = Agent::builder()
        .system_prompt(
            "あなたは添付された職務経歴書 PDF から career 情報を抽出するエージェントです。\n\
             \n\
             ユーザーメッセージに添付された PDF を読み取り、以下の手順で処理してください。\n\
             \n\
             1. PDF の内容から、各在籍企業ごとに項目を読み取り careers 配列にする。\n\
             2. careers は時系列で古い順に並べる。\n\
             3. 推測が困難な項目は空文字や \"Other\" を入れ、無理に埋めない。\n\
             \n\
             注意:\n\
             - employment_type は記載が無い場合は FullTime とする。\n\
             - detail は255文字以内厳守。\n\
             - 全項目を必ず返すこと。",
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
