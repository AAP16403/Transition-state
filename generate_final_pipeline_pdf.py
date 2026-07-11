from __future__ import annotations

import re
import textwrap
from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).resolve().parent
DETAILED = ROOT / "PSI_PIPELINE_DETAILED_PDF_READY.md"
EXACT = ROOT / "PSI_PIPELINE_EXACT_FLOW.md"
FINAL_MD = ROOT / "PSI_PIPELINE_FINAL_REPORT.md"
FINAL_PDF = ROOT / "PSI_PIPELINE_FINAL_REPORT.pdf"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_yaml(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end].strip().splitlines()
    meta: dict[str, str] = {}
    for line in raw:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[end + len("\n---") :].lstrip()


def remove_pdf_conversion_notes(text: str) -> str:
    # The final output is already a PDF, so remove instructions telling the user
    # how to convert Markdown themselves.
    text = re.sub(
        r"\nRecommended PDF conversion command when Pandoc is available:\n\n```powershell\n.*?```\n",
        "\n",
        text,
        flags=re.DOTALL,
    )
    marker = "\n# PDF Conversion Notes"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    return text


def normalize_exact_appendix(text: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    body = body.replace("# ", "## ")
    body = body.replace("## ", "### ")
    return body


def build_merged_markdown() -> str:
    detailed_meta, detailed_body = strip_yaml(read_text(DETAILED))
    detailed_body = remove_pdf_conversion_notes(detailed_body).strip()
    exact_appendix = normalize_exact_appendix(read_text(EXACT))

    title = "PSI Transition-State Pipeline Final Technical Report"
    subtitle = "Merged Detailed Flow, Cases, Equations, Algorithms, and Code-Level Audit"
    date = detailed_meta.get("date", "2026-07-07")

    merged = f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "Project PSI"
date: "{date}"
geometry: margin=1in
fontsize: 11pt
toc: true
numbersections: true
---

\\newpage

# Report Scope

This final report merges the two working technical documents:

- `PSI_PIPELINE_DETAILED_PDF_READY.md`: the main PDF-ready explanation with
  progression, cases, equations, and algorithms.
- `PSI_PIPELINE_EXACT_FLOW.md`: the code-level exact-flow audit.

The main body is organized for reading as a PDF. The appendix preserves the
exact-flow audit trail so that implementation details remain traceable to the
pipeline code.

\\newpage

{detailed_body}

\\newpage

# Appendix A: Code-Level Exact-Flow Audit

This appendix folds in the earlier exact-flow document. It intentionally keeps
some overlap with the main body because its purpose is auditability: it records
the implementation-level equations, branch behavior, and file differences in a
more direct code-reference style.

{exact_appendix}
"""
    FINAL_MD.write_text(merged, encoding="utf-8")
    return merged


class ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 16, 16)
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, self.report_title[:90], 0, 1, "R")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"Page {self.page_no()} / {{nb}}", 0, 0, "C")
        self.set_text_color(0, 0, 0)


def clean_inline_markdown(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = text.replace("\\newpage", "")
    return text


def safe_text(text: str) -> str:
    # fpdf 1.x core fonts write latin-1. The documents are intentionally ASCII,
    # but this keeps generation robust if a stray character appears later.
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b2": "^2",
        "\u00b3": "^3",
        "\u212b": "Angstrom",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def wrap_for_pdf(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    wrapped: list[str] = []
    for part in text.splitlines() or [""]:
        if part == "":
            wrapped.append("")
        else:
            wrapped.extend(
                textwrap.wrap(
                    part,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
    return wrapped


def parse_yaml_for_title(text: str) -> dict[str, str]:
    meta, _ = strip_yaml(text)
    return meta


def render_title_page(pdf: ReportPDF, meta: dict[str, str]):
    pdf.add_page()
    pdf.set_y(55)
    pdf.set_font("Arial", "B", 22)
    pdf.multi_cell(0, 11, safe_text(meta.get("title", "PSI Pipeline Report")), 0, "C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 14)
    pdf.multi_cell(0, 8, safe_text(meta.get("subtitle", "")), 0, "C")
    pdf.ln(12)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, safe_text(meta.get("author", "Project PSI")), 0, 1, "C")
    pdf.cell(0, 7, safe_text(meta.get("date", "")), 0, 1, "C")
    pdf.ln(18)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(
        0,
        6,
        safe_text(
            "Generated from the merged PSI pipeline technical report source. "
            "Equations are preserved as readable LaTeX-style notation so the "
            "PDF remains independent of external TeX/Pandoc installations."
        ),
        0,
        "C",
    )


def render_lines(pdf: ReportPDF, body: str):
    in_code = False
    in_math = False
    paragraph: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if not paragraph:
            return
        text = clean_inline_markdown(" ".join(line.strip() for line in paragraph))
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 5.2, safe_text(text))
        pdf.ln(1.2)
        paragraph = []

    def render_pre_line(line: str, math: bool = False):
        pdf.set_font("Courier", "", 8.2 if not math else 8.5)
        pdf.set_text_color(35, 35, 35)
        for wrapped in wrap_for_pdf(line, 96 if not math else 88):
            pdf.multi_cell(0, 4.2, safe_text(wrapped))

    lines = body.splitlines()

    def add_page_if_needed():
        # Avoid duplicate blank pages when the Markdown already emitted
        # \newpage immediately before a top-level heading.
        if pdf.get_y() > 28:
            pdf.add_page()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped == "\\newpage":
            flush_paragraph()
            add_page_if_needed()
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            in_code = not in_code
            if in_code:
                pdf.ln(1)
            else:
                pdf.ln(2)
            continue

        if stripped == "$$":
            flush_paragraph()
            in_math = not in_math
            if in_math:
                pdf.ln(1)
            else:
                pdf.ln(2)
            continue

        if in_code:
            render_pre_line(line)
            continue

        if in_math:
            render_pre_line(line, math=True)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("---"):
            flush_paragraph()
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            render_pre_line(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = clean_inline_markdown(heading.group(2))
            if level == 1:
                add_page_if_needed()
                pdf.set_font("Arial", "B", 16)
                pdf.set_text_color(0, 44, 82)
                pdf.multi_cell(0, 8, safe_text(text))
                pdf.ln(2)
            elif level == 2:
                pdf.set_font("Arial", "B", 13)
                pdf.set_text_color(0, 64, 104)
                pdf.multi_cell(0, 6.8, safe_text(text))
                pdf.ln(1)
            elif level == 3:
                pdf.set_font("Arial", "B", 11.5)
                pdf.set_text_color(20, 70, 100)
                pdf.multi_cell(0, 6, safe_text(text))
            else:
                pdf.set_font("Arial", "B", 10.5)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 5.5, safe_text(text))
            pdf.set_text_color(0, 0, 0)
            continue

        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            text = clean_inline_markdown(stripped[2:])
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5.2, safe_text("- " + text))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5.2, safe_text(clean_inline_markdown(stripped)))
            continue

        paragraph.append(line)

    flush_paragraph()


def render_pdf(markdown: str):
    meta, body = strip_yaml(markdown)
    title = meta.get("title", "PSI Pipeline Final Report")
    pdf = ReportPDF(title)
    render_title_page(pdf, meta)
    pdf.add_page()
    render_lines(pdf, body)
    pdf.output(str(FINAL_PDF))


def main():
    merged = build_merged_markdown()
    render_pdf(merged)
    print(f"Wrote {FINAL_MD.name} ({FINAL_MD.stat().st_size:,} bytes)")
    print(f"Wrote {FINAL_PDF.name} ({FINAL_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
