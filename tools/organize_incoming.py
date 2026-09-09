#!/usr/bin/env python3
"""Move what Gemini staged in _incoming/ into its place in the repository.

Kaggle notebooks that declare a Weights & Biases project go to that project's
code/ directory.  Everything else goes to notebooks/, grouped by topic, with a
generated index.  Checkpoints go to projects/<name>/weights/.

Run once after a scrape.  Safe to re-run: it copies, it never deletes.

    python3 tools/organize_incoming.py
"""
import csv, glob, json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INC = ROOT / "_incoming"
PROJECTS = ROOT / "projects"
NOTEBOOKS = ROOT / "notebooks"

WANDB_PROJECT = re.compile(r'project\s*=\s*["\']([^"\'/]+)["\']')

# Topic label -> keywords searched in slug and notebook source, first match wins.
TOPICS = [
    ("leakage-study-probes",
     r"split[-_]control|null[-_]distribution|identity[-_]probe|blocked[-_]cv|"
     r"leakage|luna|proxy[-_]validation|dataset[-_]audit"),
    ("lung-cancer-classification",
     r"iq[_-]?oth|lung|ccal|effnet|plos|vit[-_]|dsmat|mamba|nccd"),
    ("generative-models-for-ct",
     r"rectified[-_ ]?flow|diffusion|ddpm|aug[-_]?diff|diff[-_]aug"),
    ("gans", r"\bgan\b|dcgan|wgan|discriminator|generator|bangla.*digit"),
    ("object-detection", r"yolo|ultralytics|idcard|id[-_]card|detect|bus[-_]counter"),
    ("language-models", r"gpt|bert|transformer|tokenizer|embedding|bangla.*text"),
    ("coursework-and-practice",
     r"titanic|eda|beginner|practice|tutorial|exercise|codecademy|pandas|"
     r"movie|honey|coffee|nba|motogp|alexnet|keras"),
]


def kernel_text(d: Path) -> str:
    parts = []
    for f in d.iterdir():
        if f.suffix == ".ipynb":
            try:
                nb = json.loads(f.read_text(errors="ignore"))
            except json.JSONDecodeError:
                continue
            parts += ["".join(c.get("source", []))
                      for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        elif f.suffix in (".py", ".r", ".R"):
            parts.append(f.read_text(errors="ignore"))
    return "\n".join(parts)


def classify(slug: str, text: str) -> str:
    hay = f"{slug}\n{text[:200000]}".lower()
    for label, pattern in TOPICS:
        if re.search(pattern, hay):
            return label
    return "miscellaneous"


def kernel_meta():
    """slug -> (title, last run date) from the kernel listing pages."""
    meta = {}
    for f in sorted((INC / "kaggle").glob("kernels_page*.csv")):
        for r in csv.DictReader(f.open()):
            meta[r["ref"].split("/")[-1]] = (r["title"], r["lastRunTime"][:10])
    return meta


def move_weights():
    src = INC / "weights"
    moved = []
    for d in sorted(src.iterdir()) if src.is_dir() else []:
        files = [f for f in d.iterdir() if f.is_file()]
        if not files:
            continue
        dest = PROJECTS / d.name / "weights"
        if not dest.parent.is_dir():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").unlink(missing_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        total = sum(f.stat().st_size for f in files)
        (dest / "README.md").write_text(
            f"# weights\n\nFinal checkpoint for `{d.name}`, downloaded from the "
            f"Weights & Biases artifact store.\n\n"
            + "\n".join(f"- `{f.name}` ({f.stat().st_size/1e6:.1f} MB)" for f in files)
            + "\n\nOne checkpoint per project is kept here. Intermediate epoch "
              "checkpoints stay in Weights & Biases.\n")
        moved.append((d.name, len(files), total))
    return moved


def copy_requirements():
    """One environment capture per project, from its most recent run."""
    n = 0
    for pdir in sorted((INC / "wandb").iterdir()) if (INC / "wandb").is_dir() else []:
        dest = PROJECTS / pdir.name / "code"
        if not dest.parent.is_dir():
            continue
        reqs = sorted(pdir.glob("*/requirements.txt"), key=os.path.getmtime)
        if not reqs:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reqs[-1], dest / "requirements.txt")
        n += 1
    return n


def main():
    meta = kernel_meta()
    kernels = [d for d in sorted((INC / "kaggle").iterdir()) if d.is_dir()]
    if not kernels:
        sys.exit("nothing staged in _incoming/kaggle")

    known = {p.name for p in PROJECTS.iterdir() if p.is_dir()}
    placed, index = {}, []

    for d in kernels:
        text = kernel_text(d)
        title, date = meta.get(d.name, (d.name, ""))
        declared = {p for p in WANDB_PROJECT.findall(text) if p in known}
        if declared:
            project = sorted(declared)[0]
            dest = PROJECTS / project / "code" / "kaggle" / d.name
            placed.setdefault(project, []).append(d.name)
            where = f"projects/{project}/code/kaggle/{d.name}"
            topic = None
        else:
            topic = classify(d.name, text)
            dest = NOTEBOOKS / topic / d.name
            where = f"notebooks/{topic}/{d.name}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
        index.append({"slug": d.name, "title": title, "date": date, "where": where,
                      "topic": topic, "empty": len(text.strip()) < 400})

    write_notebook_index([i for i in index if i["topic"]])
    weights = move_weights()
    reqs = copy_requirements()

    for project, slugs in sorted(placed.items()):
        print(f"{project:32s} {len(slugs)} kernels")
    print(f"\n{sum(len(v) for v in placed.values())} kernels into projects, "
          f"{len(index) - sum(len(v) for v in placed.values())} into notebooks/")
    for name, n, size in weights:
        print(f"weights: {name} {n} file(s) {size/1e6:.0f} MB")
    print(f"requirements.txt copied for {reqs} projects")


NOTEBOOK_INTRO = """# Notebooks

{n} Kaggle notebooks that are not tied to a logged Weights & Biases project:
exploratory work, one-off probes, coursework and practice. Notebooks that produced
the runs in `projects/` live with their project instead.

Each notebook is stored exactly as it was pulled from Kaggle, alongside its
`kernel-metadata.json`. Notebooks marked *empty* were created and never filled in;
they are kept only so the record is complete.

"""


def write_notebook_index(items):
    NOTEBOOKS.mkdir(exist_ok=True)
    lines = [NOTEBOOK_INTRO.format(n=len(items))]
    for topic, _ in TOPICS + [("miscellaneous", "")]:
        group = sorted((i for i in items if i["topic"] == topic),
                       key=lambda i: i["date"], reverse=True)
        if not group:
            continue
        lines += [f"## {topic.replace('-', ' ')} ({len(group)})", "",
                  "| Notebook | Last run | |", "| --- | --- | --- |"]
        for i in group:
            note = "*empty*" if i["empty"] else ""
            lines.append(f"| [{i['title']}]({topic}/{i['slug']}/) | {i['date']} | {note} |")
        lines.append("")
    (NOTEBOOKS / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
