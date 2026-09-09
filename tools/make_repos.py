#!/usr/bin/env python3
"""Split the experiment log into one standalone GitHub repository per project.

Each output repository is self-contained: its own README, its run table, its
figures, its code and its checkpoint, with its own git history. This repository
stays behind as the index that links to all of them.

    python3 tools/make_repos.py            # writes ../github_repos/
"""
import shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as B

OUT = B.ROOT.parent / "github_repos"
USER = "tahmidulferdous"
INDEX_REPO = "ml-research-log"

# W&B project name -> (public repo name, one-line subtitle, longer description)
REPOS = {
    "lung-base-bench": (
        "lung-ct-split-leakage-benchmark",
        "How much accuracy disappears when you stop shuffling CT slices",
        "The controlled version of the leakage experiment. The same CCAL architecture, the "
        "same 70/20/10 ratio and the same three seeds are trained twice, changing only "
        "whether the slice list is shuffled before it is cut. This is the measurement the "
        "rest of the lung work builds toward, and it supersedes the larger numbers quoted "
        "in earlier write-ups."),
    "ccal_paper_reproduction": (
        "ccal-attention-cnn-reproduction",
        "Reimplementation of a concatenated CNN and attention model for lung CT",
        "A from-scratch reimplementation of the dual-branch CNN with two four-head "
        "attention blocks described in the base paper, about 666K parameters, with CLAHE "
        "preprocessing. Run under both the shuffled and the ordered split."),
    "base_ccal": (
        "ccal-attention-cnn-development",
        "Ninety runs of architecture search behind the CCAL reimplementation",
        "The development history: ninety runs of architecture and hyperparameter search "
        "that produced the model used in the reproduction repository. Kept because the "
        "search itself is the evidence of how the final configuration was reached."),
    "lung_effnet_reproduction": (
        "lung-ct-efficientnet-leakage",
        "EfficientNet B0 through B4 under both split conditions",
        "ImageNet-pretrained EfficientNet backbones from B0 to B4, 4M to 17.6M parameters, "
        "with lung-contour cropping. Each backbone is trained under both the shuffled and "
        "the ordered split so the leakage gap can be read across model capacity."),
    "vit_lung_reproduction": (
        "lung-ct-vit-leakage",
        "ViT-B/16 under both split conditions",
        "An 85.8M-parameter vision transformer on the same task, as the high-capacity end "
        "of the comparison."),
    "plos_one_lung_reproduction": (
        "lung-ct-plos-cnn-leakage",
        "A published four-block CNN, retrained under both split conditions",
        "The roughly 2M-parameter four-block CNN from the PLOS ONE paper, retrained under "
        "both splits to check whether the leakage effect is specific to one architecture."),
    "DSMaT-LungCancer": (
        "dsmat-lung-cancer-mamba",
        "Depthwise state-space (Mamba) classifiers on lung CT",
        "An exploration of depthwise separable state-space blocks as a classifier backbone "
        "for the same lung CT dataset, at Pico and Nano scales."),
    "ct-rectified-flow": (
        "ct-rectified-flow-augmentation",
        "Rectified flow generative model for CT slice augmentation",
        "A rectified-flow generator trained to synthesise CT slices, intended as a way to "
        "rebalance the heavily skewed benign class. Includes a JAX/TPU implementation "
        "alongside the PyTorch one."),
    "ct-diffusion-augmentation": (
        "ct-diffusion-augmentation",
        "Denoising diffusion model for CT slice augmentation",
        "A class-conditional DDPM trained on the same dataset for the same augmentation "
        "purpose, run before the rectified-flow work."),
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
        "YOLOv11 small and medium detectors trained to locate university ID cards, reaching "
        "0.38 mAP50-95 on the medium model. Applied work rather than research."),
    "uncategorized": (
        "early-cnn-experiments",
        "Early exploratory training runs, kept for the record",
        "Ten early runs from November 2024, before the projects above were organised. Kept "
        "so the record of what was tried is complete."),
}

GITIGNORE = "__pycache__/\n*.pyc\n.DS_Store\n.ipynb_checkpoints/\n"

CORRECTION = """
## On the split

This dataset ships no patient identifier. The filenames carry a flat slice index, not a
case number, so slices from one scan cannot be grouped. What is compared here is therefore
a **random shuffle of slices** against a **split that respects file order**, and the second
is an *ordering split*, not a patient-level split.

The distinction matters. Under a random shuffle, near-identical slices from the same scan
land on both sides of the split, and the reported accuracy is inflated by that overlap.
Holding the split ratio, the architecture and the seed fixed and changing only the shuffle
costs about **7.5 percentage points** of accuracy, measured across three seeds and two
architectures.

An earlier write-up of this work called the ordered condition patient-level and quoted a
larger gap. Both were wrong. The numbers in this repository supersede it.
"""

REPRODUCING = """
## Reproducing

Training ran on free-tier Kaggle accelerators, so the notebooks in `code/` reference
`/kaggle/input/...` paths and will not run unmodified on a local machine. To rerun, upload
`code/` as a Kaggle notebook and attach the dataset named in its header.
`code/requirements.txt` is the environment as it was captured at run time.

## Provenance

`runs.csv` is exported directly from the Weights & Biases API and is not edited by hand.
Figures are the ones the training runs logged, downscaled for size. Nothing here is
reconstructed after the fact.
"""

LUNG = {"lung-base-bench", "ccal_paper_reproduction", "base_ccal",
        "lung_effnet_reproduction", "vit_lung_reproduction",
        "plos_one_lung_reproduction", "DSMaT-LungCancer"}


def readme(project, meta, runs, repo, subtitle, description, has_weights):
    name, col = B.headline_metric(runs)
    hours = sum(B.num(r["compute_seconds"]) or 0 for r in runs) / 3600
    last = (meta["lastActive"] or meta["updatedAt"])[:10]

    out = [f"# {repo}", "", f"**{subtitle}**", "", description, "",
           "| | |", "| --- | --- |",
           f"| Runs | {len(runs)} |",
           f"| Period | {meta['createdAt'][:10]} to {last} |",
           f"| Logged compute | {hours:.1f} GPU-hours |",
           f"| Hardware | free-tier Kaggle, Tesla T4 or TPU v3-8 |",
           f"| Tracking | [Weights & Biases]({meta['wandb_url']}) |", "",
           "## What is here", "",
           "| Path | Contents |", "| --- | --- |",
           "| `runs.csv` | one row per run, every hyperparameter and every logged metric |",
           "| `figures/` | evaluation figures, grouped by run |",
           "| `code/` | the training code as it ran on Kaggle |",
           ("| `weights/` | the final checkpoint |" if has_weights else
            "| | *No checkpoint was uploaded for this project. The training "
            "code and the logged metrics are the record.* |"), "",
           "## Runs", ""]

    header = "| Run | Date | " + (f"{name} | " if name else "") + "Runtime |"
    out += [header, "| --- | --- | " + ("--- | " if name else "") + "--- |"]
    for r in sorted(runs, key=lambda r: r["created_at"]):
        val = ""
        if col:
            v = B.num(B.clean(r[col]))
            val = f"{v:.4g} | " if v is not None else "- | "
        out.append(f"| {r['display_name']} | {r['created_at'][:10]} | {val}"
                   f"{(B.num(r['compute_seconds']) or 0)/60:.0f} min |")

    if project in LUNG:
        out.append(CORRECTION)
    out.append(REPRODUCING)
    out += ["",
            f"This is one of fifteen repositories. The index, with all "
            f"{{total}} runs and the Kaggle notebooks that are not tied to a logged "
            f"project, is at "
            f"[github.com/{USER}/{INDEX_REPO}](https://github.com/{USER}/{INDEX_REPO}).",
            "",
            f"Original Weights & Biases project name: `{project}`.", "",
            "Tahmidul Bin Ferdous, Department of Computer Science and Engineering, "
            "International Islamic University Chittagong. turing.accessories@gmail.com",
            ""]
    return "\n".join(out)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    by_project = B.load_runs()
    meta = B.load_projects()
    OUT.mkdir(exist_ok=True)
    total = sum(len(v) for v in by_project.values())
    rows = []

    for project, (repo, subtitle, description) in REPOS.items():
        runs = by_project.get(project, [])
        src, dest = B.PROJECTS / project, OUT / repo
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        ckpt = [f for f in (src / "weights").glob("*")
                if f.is_file() and f.name != "README.md"] if (src / "weights").is_dir() else []
        for sub in ("figures", "code"):
            if (src / sub).is_dir() and any((src / sub).iterdir()):
                shutil.copytree(src / sub, dest / sub)
        if ckpt:
            shutil.copytree(src / "weights", dest / "weights")
        shutil.copy2(src / "runs.csv", dest / "runs.csv")
        (dest / "README.md").write_text(
            readme(project, meta[project], runs, repo, subtitle, description,
                   bool(ckpt))
            .replace("{total}", str(total)))
        (dest / ".gitignore").write_text(GITIGNORE)

        git(dest, "init", "-q")
        git(dest, "add", "-A")
        git(dest, "-c", "user.email=turing.accessories@gmail.com",
            "-c", "user.name=Tahmidul Bin Ferdous",
            "commit", "-qm", f"{subtitle}\n\n{len(runs)} runs logged to Weights & Biases "
                             f"under the project name {project}.")
        size = sum(f.stat().st_size for f in dest.rglob("*")
                   if f.is_file() and ".git/" not in str(f))
        has_w = bool(ckpt)
        rows.append((repo, project, len(runs), size, has_w))
        print(f"{repo:36s} {len(runs):3d} runs  {size/1e6:6.1f} MB"
              f"{'  +weights' if has_w else ''}")

    manifest = ["repo,wandb_project,runs,bytes,has_weights"]
    manifest += [f"{r},{p},{n},{s},{int(w)}" for r, p, n, s, w in rows]
    (OUT / "REPOS.csv").write_text("\n".join(manifest) + "\n")
    print(f"\n{len(rows)} repositories in {OUT}")


if __name__ == "__main__":
    main()
