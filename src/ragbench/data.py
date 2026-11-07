"""Loading corpora and evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

from ragbench.types import Document, EvalExample

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def load_corpus(path: str | Path) -> list[Document]:
    """Load documents from a directory of text/markdown files or a JSONL file.

    Directory: each file becomes one document whose id is the path relative to the
    directory, without suffix. JSONL: each line needs ``id`` and ``text``
    (``title`` and ``metadata`` optional).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    docs: list[Document] = []
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES:
                text = f.read_text(encoding="utf-8")
                doc_id = f.relative_to(path).with_suffix("").as_posix()
                title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), doc_id)
                docs.append(Document(id=doc_id, text=text, title=title))
    else:
        for i, row in enumerate(_read_jsonl(path)):
            if "id" not in row or "text" not in row:
                raise ValueError(f"{path}:{i + 1}: corpus rows need 'id' and 'text'")
            docs.append(
                Document(
                    id=str(row["id"]),
                    text=row["text"],
                    title=row.get("title", ""),
                    metadata=row.get("metadata", {}),
                )
            )
    if not docs:
        raise ValueError(f"No documents found in {path}")
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document ids in corpus")
    return docs


def load_dataset(path: str | Path) -> list[EvalExample]:
    path = Path(path)
    examples = []
    for i, row in enumerate(_read_jsonl(path)):
        for key in ("question", "reference_answer"):
            if key not in row:
                raise ValueError(f"{path}:{i + 1}: missing required field '{key}'")
        relevant = tuple(row.get("relevant_doc_ids", ()))
        examples.append(
            EvalExample(
                id=str(row.get("id", f"ex{i + 1}")),
                question=row["question"],
                reference_answer=row["reference_answer"],
                relevant_doc_ids=relevant,
                evidence=tuple(row.get("evidence", ())),
                answerable=bool(row.get("answerable", bool(relevant))),
                metadata=row.get("metadata", {}),
            )
        )
    ids = [e.id for e in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example ids in dataset")
    return examples


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)
