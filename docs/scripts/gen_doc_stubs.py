"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.nav.Nav()

root = Path(__file__).parent.parent.parent
src = root / "navix"
out = "api"

exclude_files = [
    "_version.py",
    "config.py"
]

for path in sorted(src.rglob("*.py")):
    if path.name in exclude_files:
        continue

    print("Generating stub for", path)
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path(out, doc_path)

    parts = tuple(module_path.parts)
    parts = ("navix",) + parts

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    
    if parts:
        nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open(f"{out}/index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
