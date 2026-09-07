# distillcourse-labs

Generated from a private monorepo; PRs and issues are not accepted. Every CI
push replaces the whole repository — a fresh tree, force-pushed, no history
guarantees.

- `labs/<course>/<lesson>/` — the student notebook (`lab.ipynb`) and the data
  files it downloads. Open a notebook in Colab:
  `https://colab.research.google.com/github/galthran-wq/distillcourse-labs/blob/main/labs/<course>/<lesson>/lab.ipynb`
- `client/` — the `distill` package the notebooks submit checkpoints with:
  `pip install "git+https://github.com/galthran-wq/distillcourse-labs#subdirectory=client"`

The lessons these labs belong to are at https://distillcourse.com.
