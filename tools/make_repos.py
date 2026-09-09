#!/usr/bin/env python3
"""Build the public GitHub repositories from the experiment log.

One repository holds the whole thesis: the four architecture families, the split
benchmark, and the generative augmentation work under extra_experiments/. The
projects that have nothing to do with the thesis get a repository each. This
repository stays behind as the index.

    python3 tools/make_repos.py            # writes ../github_repos/
"""
import shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as B

OUT = B.ROOT.parent / "github_repos"
USER = "tahmidulferdous"
INDEX_REPO = "ml-research-log"
THESIS_REPO = "lung-ct-data-leakage-study"

# --- the thesis repository -------------------------------------------------
# W&B project -> (directory inside the repo, subtitle, description)
THESIS = {
    "lung-base-bench": (
        "experiments/split-benchmark",
        "The controlled measurement: same model, same ratio, same seeds, split changed",
        "This is the headline experiment. One architecture, one 70/20/10 ratio and three "
        "seeds, trained twice, changing only whether the slice list is shuffled before it "
        "is cut. Everything else in this repository leads here."),
    "ccal_paper_reproduction": (
        "experiments/ccal-reproduction",
        "Concatenated CNN and attention model, both split conditions",
        "A from-scratch reimplementation of the dual-branch CNN with two four-head "
        "attention blocks from the base paper, about 666K parameters, with CLAHE "
        "preprocessing."),
    "base_ccal": (
        "experiments/ccal-development",
        "Ninety runs of architecture search behind the reimplementation",
        "The development history that produced the configuration used above. Kept because "
        "the search is the evidence of how that configuration was arrived at, not guessed."),
    "lung_effnet_reproduction": (
        "experiments/efficientnet",
        "EfficientNet B0 through B4, both split conditions",
        "ImageNet-pretrained EfficientNet backbones from B0 to B4, 4M to 17.6M parameters, "
        "with lung-contour cropping, so the leakage gap can be read across model capacity."),
    "vit_lung_reproduction": (
        "experiments/vit",
        "ViT-B/16, both split conditions",
        "An 85.8M-parameter vision transformer, the high-capacity end of the comparison."),
    "plos_one_lung_reproduction": (
        "experiments/plos-cnn",
        "A published four-block CNN, both split conditions",
        "The roughly 2M-parameter four-block CNN from the PLOS ONE paper, retrained to "
        "check the effect is not specific to one architecture."),
    "ct-rectified-flow": (
        "extra_experiments/rectified-flow",
        "Rectified flow generator for CT slice augmentation",
        "Work beyond the thesis. A rectified-flow generator trained to synthesise CT "
        "slices, intended to rebalance the heavily skewed benign class. Includes a "
        "JAX/TPU implementation alongside the PyTorch one."),
    "ct-diffusion-augmentation": (
        "extra_experiments/diffusion",
        "Class-conditional DDPM for CT slice augmentation",
        "Work beyond the thesis. A denoising diffusion model trained on the same dataset "
        "for the same augmentation purpose, run before the rectified-flow attempt."),
    "DSMaT-LungCancer": (
        "extra_experiments/dsmat-mamba",
        "Depthwise state-space (Mamba) classifiers on lung CT",
        "Work beyond the thesis. Depthwise separable state-space blocks as a classifier "
        "backbone on the same dataset, at Pico and Nano scales."),
}

# --- everything else, one repository each ----------------------------------
STANDALONE = {
    "BanglaGPT1": (
        "bangla-gpt-from-scratch",
        "A small GPT trained from scratch on Bengali text",
        "A decoder-only transformer written and trained from scratch on Bengali text, "
        "including the tokenizer. Built to understand the architecture rather than to "
        "compete with pretrained models."),
    "gan-bengali-digit-generation": (
        "bengali-digit-gan",
        "DCGAN and WGAN variants generating Bengali handwritten digits",
        "Twenty-eight runs comparing DCGAN, WGAN and depthwise-convolution variants on "
        "Bengali handwritten digits."),
    "WGAN-FACE": (
        "wgan-face-generation",
        "Wasserstein GAN on face images",
        "A Wasserstein GAN with gradient penalty trained on a face dataset. The generator "
        "and critic checkpoints are included."),
    "Ultralytics": (
        "yolo11-idcard-detection",
        "YOLOv11 detection for university ID cards",
        "YOLOv11 small and medium detectors trained to locate university ID cards, "
        "reaching 0.38 mAP50-95 on the medium model. Applied work rather than research."),
    "uncategorized": (
        "early-cnn-experiments",
        "Early exploratory training runs, kept for the record",
        "Ten early runs from November 2024, before the projects above were organised. "
        "Kept so the record of what was tried is complete."),
}

GITIGNORE = "__pycache__/\n*.pyc\n.DS_Store\n.ipynb_checkpoints/\n"

CORRECTION = """## What is actually being compared

The IQ-OTH/NCCD dataset ships **no patient identifier**. The filenames carry a flat slice
index, not a case number, so slices belonging to one scan cannot be grouped. Every claim
in the literature about patient-level splitting on this dataset, including an earlier
write-up of this work, is therefore unsupported by the data as released.

What can be compared, and what is compared here, is a **random shuffle of slices** against
a **split that respects file order**. The second is an *ordering split*. It is not a
patient-level split and this repository does not call it one.

The distinction is the whole point. Under a random shuffle, near-identical slices from the
same scan land on both sides of the split, and the reported accuracy is inflated by that
overlap. Holding the split ratio, the architecture and the seed fixed and changing only the
shuffle costs about **7.5 percentage points** of accuracy, measured across three seeds and
two architectures.

An earlier write-up called the ordered condition patient-level and quoted a gap of 12.46
points. Both figures were wrong: the first conflated grouping with a change in split ratio,
and the second described a split the code never performed. The numbers in this repository
supersede them.
"""

REPRODUCING = """## Reproducing

Training ran on free-tier Kaggle accelerators, so the notebooks reference `/kaggle/input/...`
paths and will not run unmodified on a local machine. To rerun an experiment, upload its
`code/` directory as a Kaggle notebook and attach the dataset named in the header.
Each `code/requirements.txt` is the environment as captured at run time.

## Provenance

Every `runs.csv` is exported directly from the Weights & Biases API and is not edited by
hand. Figures are the ones the training runs logged, downscaled for size. Nothing here is
reconstructed after the fact.
"""


def facts_table(meta, runs):
    hours = sum(B.num(r["compute_seconds"]) or 0 for r in runs) / 3600
    last = (meta["lastActive"] or meta["updatedAt"])[:10]
    return ["| | |", "| --- | --- |",
            f"| Runs | {len(runs)} |",
            f"| Period | {meta['createdAt'][:10]} to {last} |",
            f"| Logged compute | {hours:.1f} GPU-hours |",
            f"| Hardware | free-tier Kaggle, Tesla T4 or TPU v3-8 |",
            f"| Tracking | [Weights & Biases]({meta['wandb_url']}) |", ""]


def run_table(runs):
    name, col = B.headline_metric(runs)
    out = ["| Run | Date | " + (f"{name} | " if name else "") + "Runtime |",
           "| --- | --- | " + ("--- | " if name else "") + "--- |"]
    for r in sorted(runs, key=lambda r: r["created_at"]):
        val = ""
        if col:
            v = B.num(B.clean(r[col]))
            val = f"{v:.4g} | " if v is not None else "- | "
        out.append(f"| {r['display_name']} | {r['created_at'][:10]} | {val}"
                   f"{(B.num(r['compute_seconds']) or 0)/60:.0f} min |")
    return out + [""]


def contents_table(has_weights):
    return ["| Path | Contents |", "| --- | --- |",
            "| `runs.csv` | one row per run, every hyperparameter and every logged metric |",
            "| `figures/` | evaluation figures, grouped by run |",
            "| `code/` | the training code as it ran on Kaggle |",
            ("| `weights/` | the final checkpoint |" if has_weights else
             "| | *No checkpoint was uploaded for these runs. The code and the logged "
             "metrics are the record.* |"), ""]


def copy_experiment(project, dest):
    """Copy one project's material into dest. Returns True if a checkpoint came along."""
    src = B.PROJECTS / project
    ckpt = [f for f in (src / "weights").glob("*")
            if f.is_file() and f.name != "README.md"] if (src / "weights").is_dir() else []
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "code"):
        if (src / sub).is_dir() and any((src / sub).iterdir()):
            shutil.copytree(src / sub, dest / sub, dirs_exist_ok=True)
    if ckpt:
        shutil.copytree(src / "weights", dest / "weights", dirs_exist_ok=True)
    shutil.copy2(src / "runs.csv", dest / "runs.csv")
    return bool(ckpt)


def sub_readme(project, meta, runs, subtitle, description, has_weights, up):
    return "\n".join(
        [f"# {subtitle}", "", description, ""]
        + facts_table(meta, runs)
        + ["## What is here", ""] + contents_table(has_weights)
        + ["## Runs", ""] + run_table(runs)
        + [f"Part of [{THESIS_REPO}]({up}). "
           f"Original Weights & Biases project name: `{project}`.", ""])


def build_thesis(by_project, meta, total_runs):
    dest = OUT / THESIS_REPO
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    rows, any_weights = [], False
    for project, (path, subtitle, description) in THESIS.items():
        runs = by_project.get(project, [])
        d = dest / path
        has_w = copy_experiment(project, d)
        any_weights |= has_w
        up = "../.." if path.count("/") == 1 else "../../.."
        (d / "README.md").write_text(
            sub_readme(project, meta[project], runs, subtitle, description, has_w, up))
        rows.append((path, subtitle, len(runs), has_w))

    lead = by_project["lung-base-bench"]
    _, col = B.headline_metric(lead)
    seeds, pairs = {}, []
    for r in lead:
        seed = r["display_name"].rsplit("seed", 1)[-1]
        kind = "shuffled" if "shuffled" in r["display_name"] else "ordered"
        seeds.setdefault(seed, {})[kind] = B.num(B.clean(r[col]))
    for seed in sorted(seeds, key=int):
        v = seeds[seed]
        if v.get("shuffled") and v.get("ordered"):
            pairs.append((seed, v["shuffled"], v["ordered"]))
    mean_s = sum(p[1] for p in pairs) / len(pairs)
    mean_o = sum(p[2] for p in pairs) / len(pairs)

    out = [
        f"# {THESIS_REPO}",
        "",
        "**What a lung CT classifier's accuracy really is, once you stop shuffling "
        "slices between train and test**",
        "",
        "Undergraduate thesis work and everything around it: the experiments, the code as "
        "it ran, the logged metrics for every run, and the figures those runs produced. "
        f"{sum(len(by_project.get(p, [])) for p in THESIS)} training runs in total, all on "
        "free-tier Kaggle accelerators.",
        "",
        "## The result",
        "",
        "| Seed | Shuffled split | File-order split | Difference |",
        "| --- | --- | --- | --- |",
    ] + [
        f"| {seed} | {s_:.2f} | {o:.2f} | {o - s_:+.2f} |" for seed, s_, o in pairs
    ] + [
        f"| **mean** | **{mean_s:.2f}** | **{mean_o:.2f}** | "
        f"**{mean_o - mean_s:+.2f}** |",
        "",
        "Test accuracy, percent. Same architecture, same 70/20/10 ratio, same seeds. The "
        "only thing that changed is whether the slice list was shuffled before it was cut. "
        "Full per-run numbers are in "
        "[experiments/split-benchmark/runs.csv](experiments/split-benchmark/runs.csv).",
        "",
        CORRECTION,
        "## The experiments",
        "",
        "| Experiment | Runs | What it is |",
        "| --- | --- | --- |",
    ]
    for path, subtitle, n, _ in rows:
        if path.startswith("experiments/"):
            out.append(f"| [{path.split('/')[-1]}]({path}/) | {n} | {subtitle} |")
    out += ["", "## Beyond the thesis", "",
            "Attempts to fix the class imbalance by generating CT slices, and one "
            "architecture exploration. None of these fed into the thesis result; they are "
            "here because they were done.", "",
            "| Experiment | Runs | What it is |", "| --- | --- | --- |"]
    for path, subtitle, n, _ in rows:
        if path.startswith("extra_experiments/"):
            out.append(f"| [{path.split('/')[-1]}]({path}/) | {n} | {subtitle} |")

    out += ["", "## Layout", "",
            "```",
            "experiments/          the four architecture families and the split benchmark",
            "extra_experiments/    generative augmentation and one architecture probe",
            "```",
            "",
            "Every experiment directory holds the same four things: `runs.csv`, `figures/`, "
            "`code/`, and `weights/` where a checkpoint was saved.",
            "",
            REPRODUCING,
            f"The wider record, all {total_runs} runs across every project plus 162 Kaggle "
            f"notebooks, is at [github.com/{USER}/{INDEX_REPO}]"
            f"(https://github.com/{USER}/{INDEX_REPO}).",
            "",
            "Tahmidul Bin Ferdous, Department of Computer Science and Engineering, "
            "International Islamic University Chittagong. turing.accessories@gmail.com",
            ""]
    (dest / "README.md").write_text("\n".join(out))
    (dest / ".gitignore").write_text(GITIGNORE)
    return dest, sum(len(by_project.get(p, [])) for p in THESIS), any_weights


def build_standalone(project, repo, subtitle, description, meta, runs, total_runs):
    dest = OUT / repo
    if dest.exists():
        shutil.rmtree(dest)
    has_w = copy_experiment(project, dest)
    out = ([f"# {repo}", "", f"**{subtitle}**", "", description, ""]
           + facts_table(meta, runs)
           + ["## What is here", ""] + contents_table(has_w)
           + ["## Runs", ""] + run_table(runs)
           + [REPRODUCING,
              f"One of seven repositories. The index, with all {total_runs} runs and the "
              f"Kaggle notebooks that are not tied to a logged project, is at "
              f"[github.com/{USER}/{INDEX_REPO}](https://github.com/{USER}/{INDEX_REPO}).",
              "",
              f"Original Weights & Biases project name: `{project}`.", "",
              "Tahmidul Bin Ferdous, Department of Computer Science and Engineering, "
              "International Islamic University Chittagong. turing.accessories@gmail.com",
              ""])
    (dest / "README.md").write_text("\n".join(out))
    (dest / ".gitignore").write_text(GITIGNORE)
    return dest, has_w


def git_init(repo, message):
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=turing.accessories@gmail.com",
                  "-c", "user.name=Tahmidul Bin Ferdous", "commit", "-qm", message]):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def size_of(d):
    return sum(f.stat().st_size for f in d.rglob("*")
               if f.is_file() and ".git/" not in str(f))


def main():
    by_project = B.load_runs()
    meta = B.load_projects()
    OUT.mkdir(exist_ok=True)
    total = sum(len(v) for v in by_project.values())
    rows = []

    d, n, has_w = build_thesis(by_project, meta, total)
    git_init(d, "Lung CT data leakage study: thesis experiments and code\n\n"
                f"{n} runs across nine Weights & Biases projects.")
    rows.append((THESIS_REPO, "thesis", n, size_of(d), has_w))
    print(f"{THESIS_REPO:34s} {n:3d} runs  {size_of(d)/1e6:6.1f} MB  "
          f"{len(THESIS)} experiments")

    for project, (repo, subtitle, description) in STANDALONE.items():
        runs = by_project.get(project, [])
        d, has_w = build_standalone(project, repo, subtitle, description,
                                    meta[project], runs, total)
        git_init(d, f"{subtitle}\n\n{len(runs)} runs logged to Weights & Biases "
                    f"under the project name {project}.")
        rows.append((repo, project, len(runs), size_of(d), has_w))
        print(f"{repo:34s} {len(runs):3d} runs  {size_of(d)/1e6:6.1f} MB"
              f"{'  +weights' if has_w else ''}")

    (OUT / "REPOS.csv").write_text(
        "\n".join(["repo,wandb_project,runs,bytes,has_weights"]
                  + [f"{r},{p},{n},{s},{int(w)}" for r, p, n, s, w in rows]) + "\n")
    print(f"\n{len(rows)} repositories in {OUT}")


if __name__ == "__main__":
    main()
