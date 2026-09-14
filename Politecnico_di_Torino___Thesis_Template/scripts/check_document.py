"""Check the built document and prepare page contact sheets for visual review."""
from pathlib import Path
import json
import re
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "build/qa"
QA.mkdir(parents=True, exist_ok=True)
PDF = ROOT / "build/thesis.pdf"
subprocess.run(["pdftotext", "-layout", str(PDF), str(QA / "thesis.txt")], check=True)
subprocess.run(["pdftoppm", "-r", "65", "-png", str(PDF), str(QA / "page")], check=True)
pages = (QA / "thesis.txt").read_text(encoding="utf-8").split("\f")
if not pages[-1].strip():
    pages.pop()

visited = {}


def source(path):
    path = path.resolve()
    if path in visited:
        return
    raw = path.read_text(encoding="utf-8-sig")
    clean = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in raw.splitlines())
    visited[path] = clean
    for included in re.findall(r"\\input\{([^}]+)\}", clean):
        child = ROOT / included
        if not child.suffix:
            child = child.with_suffix(".tex")
        source(child)


source(ROOT / "thesis.tex")
combined = "\n".join(visited.values())
bib = (ROOT / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", combined) for key in group.split(",")}
labels = re.findall(r"\\label\{([^}]+)\}", combined)
references = re.findall(r"\\ref\{([^}]+)\}", combined)
log = (ROOT / "build/thesis.log").read_text(encoding="utf-8", errors="replace")
warnings = [line for line in log.splitlines() if "Warning:" in line or "Overfull" in line]
report = {
    "pdf_pages": len(pages),
    "approximate_extracted_words_including_front_matter_and_references": len(re.findall(r"\b[\w'-]+\b", "\n".join(pages))),
    "compiled_source_files": len(visited),
    "bibliography_entries": len(keys),
    "cited_entries": len(citations),
    "missing_citations": sorted(citations - keys),
    "missing_cross_references": sorted(set(references) - set(labels)),
    "duplicate_labels": sorted({key for key in labels if labels.count(key) > 1}),
    "unresolved_markers": [i + 1 for i, p in enumerate(pages) if "??" in p],
    "very_short_pages": [i + 1 for i, p in enumerate(pages) if len(p.split()) < 15],
    "log_warnings": warnings,
    "page_openings": {i + 1: " ".join(p.split()[:14]) for i, p in enumerate(pages)},
}
for group_start in range(0, len(pages), 9):
    sheet = Image.new("RGB", (1050, 1554), "#d9dfe4")
    draw = ImageDraw.Draw(sheet)
    for offset in range(min(9, len(pages)-group_start)):
        number = group_start + offset + 1
        candidates = sorted(QA.glob(f"page-{number:02d}.png"))
        if not candidates:
            continue
        with Image.open(candidates[0]) as page:
            page.thumbnail((330, 472))
            col, row = offset % 3, offset // 3
            x, y = 10 + col*350, 28 + row*518
            sheet.paste(page, (x, y))
            draw.text((x, y-19), f"PDF page {number}", fill="#172c3c")
    sheet.save(QA / f"contact-{group_start+1:02d}.png")

(QA / "document_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "page_openings"}, indent=2))
print(json.dumps(report["page_openings"], indent=2))
assert not report["missing_citations"]
assert not report["missing_cross_references"]
assert not report["duplicate_labels"]
assert not report["unresolved_markers"]
