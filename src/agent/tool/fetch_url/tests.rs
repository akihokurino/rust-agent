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
