#!/usr/bin/env python3
"""Screen thesis prose for unusually long overlap with cited source texts.

This is a reproducible similarity screen, not a plagiarism determination.  It
compares the authored LaTeX prose (excluding generated result tables) with
plain text extracted from source PDFs and reports exact long spans plus highly
similar sentence pairs for human review.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def remove_balanced_command(text: str, command: str) -> str:
    pattern = re.compile(r"\\" + re.escape(command) + r"(?:\[[^\]]*\])?\{")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{" and text[index - 1] != "\\":
                depth += 1
            elif text[index] == "}" and text[index - 1] != "\\":
                depth -= 1
            index += 1
        text = text[: match.start()] + " " + text[index:]


def latex_to_text(raw: str) -> str:
    raw = re.sub(r"(?<!\\)%.*", " ", raw)
    for command in ("cite", "parencite", "textcite", "autocite", "ref", "pageref", "label"):
        raw = remove_balanced_command(raw, command)
    raw = re.sub(r"\\begin\{(?:table|figure|equation|align|lstlisting)[^}]*\}.*?\\end\{[^}]+\}", " ", raw, flags=re.S)
    raw = re.sub(r"\$[^$]*\$", " ", raw, flags=re.S)
    raw = re.sub(r"\\\[[\s\S]*?\\\]", " ", raw)
    raw = re.sub(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}", " ", raw)
    raw = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", raw)
    raw = raw.replace("~", " ").replace("\\&", "and").replace("--", "-")
    raw = raw.replace("{", " ").replace("}", " ")
    raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if len(tokens(part)) >= 12]


def load_thesis(root: Path) -> dict[str, str]:
    content = root / "content"
    paths = [
        content / "abstract.tex",
        content / "summary.tex",
        content / "acknowledgements.tex",
        *(sorted((content / "chapters").glob("chapter*.tex"))),
        content / "appendix.tex",
    ]
    return {str(path.relative_to(root)): latex_to_text(path.read_text(encoding="utf-8")) for path in paths}


def load_sources(source_dir: Path) -> dict[str, str]:
    return {
        path.stem: re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="ignore")).strip()
        for path in sorted(source_dir.glob("*.txt"))
    }


def exact_matches(thesis: dict[str, str], sources: dict[str, str], minimum: int) -> list[dict[str, object]]:
    source_tokens = {name: tokens(text) for name, text in sources.items()}
    index: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for name, words in source_tokens.items():
        for pos in range(len(words) - minimum + 1):
            index[tuple(words[pos : pos + minimum])].append((name, pos))

    findings: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for thesis_file, text in thesis.items():
        thesis_tokens = tokens(text)
        for thesis_pos in range(len(thesis_tokens) - minimum + 1):
            key = tuple(thesis_tokens[thesis_pos : thesis_pos + minimum])
            for source, source_pos in index.get(key, []):
                source_words = source_tokens[source]
                if thesis_pos and source_pos and thesis_tokens[thesis_pos - 1] == source_words[source_pos - 1]:
                    continue
                length = minimum
                while (
                    thesis_pos + length < len(thesis_tokens)
                    and source_pos + length < len(source_words)
                    and thesis_tokens[thesis_pos + length] == source_words[source_pos + length]
                ):
                    length += 1
                identity = (thesis_file, source, thesis_pos, length)
                if identity not in seen:
                    seen.add(identity)
                    findings.append(
                        {
                            "thesis_file": thesis_file,
                            "source": source,
                            "word_count": length,
                            "text": " ".join(thesis_tokens[thesis_pos : thesis_pos + length]),
                        }
                    )
    findings.sort(key=lambda item: (-int(item["word_count"]), str(item["thesis_file"])))
    return findings


def similar_sentences(thesis: dict[str, str], sources: dict[str, str], threshold: float) -> list[dict[str, object]]:
    source_sentences = {name: sentences(text) for name, text in sources.items()}
    source_index: dict[tuple[str, ...], set[tuple[str, int]]] = defaultdict(set)
    for source, items in source_sentences.items():
        for index, sentence in enumerate(items):
            words = tokens(sentence)
            for pos in range(max(0, len(words) - 4)):
                source_index[tuple(words[pos : pos + 5])].add((source, index))

    findings: list[dict[str, object]] = []
    for thesis_file, text in thesis.items():
        for thesis_sentence in sentences(text):
            thesis_words = tokens(thesis_sentence)
            candidates: set[tuple[str, int]] = set()
            for pos in range(max(0, len(thesis_words) - 4)):
                candidates.update(source_index.get(tuple(thesis_words[pos : pos + 5]), set()))
            for source, index in candidates:
                source_sentence = source_sentences[source][index]
                source_words = tokens(source_sentence)
                length_ratio = min(len(thesis_words), len(source_words)) / max(len(thesis_words), len(source_words))
                if length_ratio < 0.55:
                    continue
                score = SequenceMatcher(None, thesis_words, source_words, autojunk=False).ratio()
                if score >= threshold:
                    findings.append(
                        {
                            "thesis_file": thesis_file,
                            "source": source,
                            "similarity": round(score, 3),
                            "thesis_sentence": thesis_sentence,
                            "source_sentence": source_sentence,
                        }
                    )
    findings.sort(key=lambda item: (-float(item["similarity"]), str(item["thesis_file"])))
    return findings


def write_markdown(path: Path, report: dict[str, object]) -> None:
    exact = report["exact_matches"]
    similar = report["similar_sentences"]
    lines = [
        "# Originality overlap screen",
        "",
        "This automated report is a source-overlap aid, not a plagiarism verdict and not a substitute for the university's approved checker.",
        "",
        f"- Thesis files screened: {report['thesis_file_count']}",
        f"- Cited source texts screened: {report['source_count']}",
        f"- Exact matches of at least {report['minimum_exact_words']} words: {len(exact)}",
        f"- Sentence pairs at or above {report['sentence_similarity_threshold']:.0%} similarity: {len(similar)}",
        "",
        "## Exact matches",
        "",
    ]
    if not exact:
        lines.append("No long exact matches found.")
    for item in exact:
        lines.extend(
            [
                f"- **{item['word_count']} words** — `{item['thesis_file']}` vs. `{item['source']}`",
                f"  - {item['text']}",
            ]
        )
    lines.extend(["", "## Similar sentences", ""])
    if not similar:
        lines.append("No highly similar sentence pairs found.")
    for item in similar:
        lines.extend(
            [
                f"- **{item['similarity']:.1%}** — `{item['thesis_file']}` vs. `{item['source']}`",
                f"  - Thesis: {item['thesis_sentence']}",
                f"  - Source: {item['source_sentence']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-exact-words", type=int, default=12)
    parser.add_argument("--sentence-similarity-threshold", type=float, default=0.78)
    args = parser.parse_args()

    root = args.root.resolve()
    source_dir = (args.source_dir or root / "build" / "qa" / "originality" / "sources").resolve()
    output_dir = (args.output_dir or root / "build" / "qa" / "originality").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    thesis = load_thesis(root)
    sources = load_sources(source_dir)
    if not sources:
        raise SystemExit(f"No source .txt files found in {source_dir}")

    report = {
        "thesis_file_count": len(thesis),
        "source_count": len(sources),
        "minimum_exact_words": args.minimum_exact_words,
        "sentence_similarity_threshold": args.sentence_similarity_threshold,
        "exact_matches": exact_matches(thesis, sources, args.minimum_exact_words),
        "similar_sentences": similar_sentences(thesis, sources, args.sentence_similarity_threshold),
    }
    (output_dir / "originality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(output_dir / "originality_report.md", report)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}, indent=2))
    print(f"exact_matches={len(report['exact_matches'])}")
    print(f"similar_sentences={len(report['similar_sentences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
