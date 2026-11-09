"""Render the example notebooks as Markdown pages for the documentation site.

    uv run --group docs python scripts/notebooks_to_docs.py

Each ``examples/<name>.ipynb`` becomes ``docs/examples/<name>.md`` (plus its images), with the
saved outputs and a link to the notebook on GitHub. The generated files are git-ignored: the
notebook stays the only source. Run this before ``zensical build`` or ``zensical serve``.
"""

import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "examples"
TARGET = ROOT / "docs" / "examples"
GITHUB = "https://github.com/nicolasambin/lazycontroller/blob/main/examples"

HEADER = """\
!!! note "This page is a notebook"
    It was run once and saved with its outputs.
    [Open it on GitHub]({url}) or download it to run it yourself.

"""


def convert(notebook: Path) -> Path:
    body, resources = MarkdownExporter().from_notebook_node(
        nbformat.read(notebook, as_version=4),
        resources={"output_files_dir": f"{notebook.stem}_files"},
    )
    # The note goes right under the notebook's own title, which becomes the page title.
    title, _, rest = body.partition("\n")
    header = HEADER.format(url=f"{GITHUB}/{notebook.name}")
    page = TARGET / f"{notebook.stem}.md"
    page.write_text(f"{title}\n\n{header}{rest}" if title.startswith("# ") else header + body)
    for name, data in resources.get("outputs", {}).items():
        (TARGET / name).parent.mkdir(parents=True, exist_ok=True)
        (TARGET / name).write_bytes(data)
    return page


def main() -> None:
    shutil.rmtree(TARGET, ignore_errors=True)
    TARGET.mkdir(parents=True)
    for notebook in sorted(SOURCE.glob("*.ipynb")):
        print(f"{notebook.relative_to(ROOT)} -> {convert(notebook).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
