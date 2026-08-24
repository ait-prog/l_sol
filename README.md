# Lost in the Museum — solution

Kaggle competition: cross-domain image retrieval. Generate a fixed-dimensional,
L2-normalized embedding for each of 20,000 images (10,000 HQ gallery paintings,
1,000 real-world visitor-photo queries, 9,000 distractor "dummy" images) so that
a query's embedding is closest (cosine similarity) to its true HQ match, scored
via Hit@3.

## Contents

- `notebooks/lost_in_the_museum_clip_baseline.ipynb` — baseline solution.
  Extracts CLIP (`openai/clip-vit-base-patch32`, 512-d) image embeddings for
  every image, L2-normalizes them, and writes `submission.csv` in the required
  format.

## How to run

This notebook is meant to run **inside a Kaggle Notebook**, not locally — the
dataset (~10 GB) is already mounted there under `/kaggle/input/...`, so no
download is needed, and the competition requires a working notebook to be
submitted for reproducibility anyway.

1. On the competition page, click **New Notebook** (or copy the cells from
   `notebooks/lost_in_the_museum_clip_baseline.ipynb` into a new Kaggle
   notebook).
2. Attach the competition dataset if it isn't already attached.
3. In Settings: turn on an **accelerator** (GPU T4 x2 or P100) and make sure
   **Internet** is on (needed once, to download the pretrained CLIP weights).
4. Run all cells. The notebook auto-discovers the dataset layout at runtime
   (it doesn't hardcode any path) and validates itself against the provided
   `submission.csv` template before writing the final output.
5. Submit the resulting `submission.csv`.

If the auto-discovery step reports missing files, duplicate filenames, or a
dimension mismatch, paste that diagnostic output back for a targeted fix
rather than guessing at the dataset layout from scratch.

See the notebook's final markdown cell for ideas on improving the Hit@3 score
further (bigger backbone, ensembling with DINOv2, stronger TTA, embedding
post-processing).
