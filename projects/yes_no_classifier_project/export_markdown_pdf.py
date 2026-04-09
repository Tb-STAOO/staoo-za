#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer


def build_styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=11,
        leading=16,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=base,
        fontSize=20,
        leading=24,
        spaceBefore=8,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=base,
        fontSize=16,
        leading=20,
        spaceBefore=6,
        spaceAfter=8,
    )
    h3 = ParagraphStyle(
        "H3CN",
        parent=base,
        fontSize=13,
        leading=18,
        spaceBefore=4,
        spaceAfter=6,
    )
    code = ParagraphStyle(
        "CodeCN",
        parent=base,
        fontName="Courier",
        fontSize=9,
        leading=12,
        backColor="#F5F5F5",
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=8,
    )
    table = ParagraphStyle(
        "TableCN",
        parent=code,
        fontName="STSong-Light",
        fontSize=9.5,
        leading=12,
    )
    return {"body": base, "h1": h1, "h2": h2, "h3": h3, "code": code, "table": table}


def md_to_story(md_path: Path, doc_width: float):
    styles = build_styles()
    story: List = []

    lines = md_path.read_text(encoding="utf-8").splitlines()
    para_buf: List[str] = []
    code_buf: List[str] = []
    table_buf: List[str] = []
    in_code = False

    def flush_para():
        if not para_buf:
            return
        text = " ".join(s.strip() for s in para_buf if s.strip())
        if text:
            story.append(Paragraph(html.escape(text), styles["body"]))
        para_buf.clear()

    def flush_code():
        if not code_buf:
            return
        story.append(Preformatted("\n".join(code_buf), styles["code"]))
        code_buf.clear()

    def flush_table():
        if not table_buf:
            return
        story.append(Preformatted("\n".join(table_buf), styles["table"]))
        table_buf.clear()

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            flush_table()
            in_code = not in_code
            if not in_code:
                flush_code()
            continue

        if in_code:
            code_buf.append(line)
            continue

        if stripped.startswith("|"):
            flush_para()
            table_buf.append(line)
            continue
        flush_table()

        if not stripped:
            flush_para()
            story.append(Spacer(1, 3))
            continue

        m = re.match(r"!\[[^\]]*]\(([^)]+)\)", stripped)
        if m:
            flush_para()
            img_rel = m.group(1).strip()
            img_path = (md_path.parent / img_rel).resolve()
            if img_path.exists():
                iw, ih = ImageReader(str(img_path)).getSize()
                if iw > 0 and ih > 0:
                    width = min(doc_width, float(iw))
                    height = width * float(ih) / float(iw)
                    if height > 220 * mm:
                        scale = (220 * mm) / height
                        width *= scale
                        height *= scale
                    story.append(Image(str(img_path), width=width, height=height))
                    story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(html.escape(f"[Missing image] {img_rel}"), styles["body"]))
            continue

        if stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                style = styles["h1"]
            elif level == 2:
                style = styles["h2"]
            else:
                style = styles["h3"]
            story.append(Paragraph(html.escape(text), style))
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            bullet_text = stripped[2:].strip()
            story.append(Paragraph(html.escape(f"• {bullet_text}"), styles["body"]))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            story.append(Paragraph(html.escape(stripped), styles["body"]))
            continue

        para_buf.append(stripped)

    flush_para()
    flush_table()
    flush_code()
    return story


def convert(md_path: Path, pdf_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=md_path.stem,
    )
    story = md_to_story(md_path, doc.width)
    doc.build(story)


def parse_args():
    parser = argparse.ArgumentParser(description="Export markdown report to PDF (Chinese supported).")
    parser.add_argument(
        "--input",
        type=str,
        default="REPORT_MAIN_PDF_STYLE.md",
        help="Input markdown path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="REPORT_MAIN_PDF_STYLE.pdf",
        help="Output PDF path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    md_path = Path(args.input).resolve()
    pdf_path = Path(args.output).resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"Input markdown not found: {md_path}")
    convert(md_path, pdf_path)
    print(f"PDF generated: {pdf_path}")


if __name__ == "__main__":
    main()
