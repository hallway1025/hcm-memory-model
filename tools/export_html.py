#!/usr/bin/env python3
"""export_html.py —— PAPER_final.md -> 单文件可打印 HTML（无外部依赖）。"""
import os, re, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "PAPER_final.md")
OUT = os.path.join(ROOT, "paper", "HCM_PAPER.html")
IMG = re.compile(r"!\[(.*?)\]\((.*?)\)")

CSS = """
body{max-width:860px;margin:32px auto;padding:0 20px;font-family:"Microsoft YaHei",system-ui,sans-serif;
line-height:1.75;color:#1a1a1a;font-size:15px}
h1{font-size:26px;text-align:center;margin:8px 0 20px}
h2{font-size:20px;margin:30px 0 10px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:16px;margin:22px 0 8px}
p{text-align:justify;margin:8px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px}
th,td{border:1px solid #bbb;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f2f2f2}
img{max-width:100%;display:block;margin:14px auto}
figure{text-align:center;font-size:12.5px;color:#444}
pre{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:4px;padding:10px;overflow-x:auto}
code{font-family:Consolas,monospace;font-size:12.5px}
hr{border:0;border-top:1px solid #ddd;margin:26px 0}
blockquote{color:#555;border-left:3px solid #ccc;margin:10px 0;padding-left:12px}
ul,ol{margin:8px 0 8px 22px}
"""


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\bdoi:(10\.\S+?)(?=[\s<]|$)", r'doi:<a href="https://doi.org/\1">\1</a>', s)
    return s


def main() -> int:
    import sys as _sys
    src = _sys.argv[1] if len(_sys.argv) > 1 else SRC
    outp = _sys.argv[2] if len(_sys.argv) > 2 else OUT
    lang = "en" if "EN" in os.path.basename(src) else "zh-CN"
    lines = open(src, encoding="utf-8").read().split("\n")

    # PDF /Title 元数据：从源文件首个标题行取，取不到才回退。
    # 早期这里硬编码 "HCM paper" / "HCM 记忆模型论文"，每次重渲都会把
    # 论文正式标题覆盖回那个占位串（v1.3 的 9553bba 修过又被覆盖）。
    # 兼容三种写法：'# 标题' / BOM 前缀 '# 标题' / 首行裸标题（EN 稿如此）。
    title = None
    for _ln in lines:
        _clean = _ln.lstrip("\ufeff").strip()
        if not _clean:
            continue
        if _clean.startswith("# "):
            title = _clean[2:].strip()
        elif not _clean.startswith(("#", ">", "|", "**", "---", "```", "~~~")):
            # 首个实质内容是普通文本行 → 视为标题（EN 主稿的首行裸标题）
            title = _clean
        break
    if not title:
        title = "HCM paper" if lang == "en" else "HCM 记忆模型论文"
    # 去掉 markdown 强调符，避免元数据里混入 **
    title = title.replace("**", "").replace("*", "")
    out, i, in_code, in_ul, in_ol = [], 0, False, False, False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("~~~"):
            close_lists()
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(html.escape(ln))
            i += 1
            continue
        if not ln.strip():
            # 空行不立即关闭列表：markdown 有序列表项之间允许空行（松散列表），
            # 若立刻 close_lists，每个列表项都会变成从 1 重编号的独立 <ol>。
            # 只有当下一行不是列表项延续时才关闭。
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not (in_ul and re.match(r"^[-*]\s+", nxt)) and not (in_ol and re.match(r"^\d+\.\s+", nxt)):
                close_lists()
            i += 1
            continue
        m = IMG.fullmatch(ln.strip())
        if m:
            close_lists()
            out.append('<figure><img src="%s" alt="%s">' % (m.group(2), html.escape(m.group(1))))
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*图"):
                out.append("<figcaption>%s</figcaption>" % inline(lines[i + 1].strip().strip("*")))
                i += 1
            out.append("</figure>")
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            close_lists()
            out.append("<table>")
            first = True
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not re.match(r"^[\s:\-]+$", "".join(cells)):
                    tag = "th" if first else "td"
                    out.append("<tr>" + "".join("<%s>%s</%s>" % (tag, inline(c), tag) for c in cells) + "</tr>")
                    first = False
                i += 1
            out.append("</table>")
            continue
        if ln.startswith("#"):
            close_lists()
            lvl = min(4, len(ln) - len(ln.lstrip("#")))
            out.append("<h%d>%s</h%d>" % (lvl, inline(ln[lvl:].strip()), lvl))
            i += 1
            continue
        if ln.strip() == "---":
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        m2 = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if m2:
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            out.append("<li>%s</li>" % inline(m2.group(1)))
            i += 1
            continue
        m3 = re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
        if m3:
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            out.append("<li>%s</li>" % inline(m3.group(2)))
            i += 1
            continue
        close_lists()
        out.append("<p>%s</p>" % inline(ln.strip()))
        i += 1

    doc = ("<!doctype html><html lang=\"%s\"><meta charset=\"utf-8\">"
           "<title>%s</title><style>%s</style><body>%s</body></html>") % (lang, title, CSS, "\n".join(out))
    open(outp, "w", encoding="utf-8").write(doc)
    print("wrote", outp, len(doc) // 1024, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
