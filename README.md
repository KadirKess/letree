# Lεtree

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KadirKess/letree/blob/main/notebooks/train.ipynb)

Lεtree learns the ε-tree of a noisy image, a hierarchical structure over its
regions that estimates the component tree of the clean image from the noisy
image alone. The name reads "Le-tree", for Learning the ε-tree, where ε is the
noise term in y = x + ε. Where Greek letters cannot appear, it is written
`letree`.

A component tree such as the max-tree orders the flat zones of a grayscale image
(connected regions of constant gray level) by inclusion. Noise splits an image
into many small flat zones that the clean image does not have, so the max-tree
of a noisy image mostly describes the noise. Lεtree starts from the region
adjacency graph (RAG) of the noisy image, in which each node is a flat zone and
edges join zones that touch. A graph neural network predicts, for every zone, a
corrective parent and an action that either merges the zone into that parent or
keeps it as a node at the same depth as the parent or one step below it.
Decoding these predictions groups the noisy zones and links the groups into the
ε-tree.

The ε-tree is a tree, but it is neither the clean max-tree nor a hierarchy of
partitions. Its nodes are groups of observed zones that need not be connected,
a clean region that noise split apart can stay split across several groups, and
its depths count changes of gray level rather than storing the gray levels
themselves. Training targets are derived from the clean component tree, and the
scores compare the decoded ε-tree with the clean image by lost zones,
misassigned pixels and depth error. This is research code for a paper in
preparation, and only a preliminary salt-and-pepper model has been trained so
far.

## Layout

```
src/letree/
  component_tree/   RAG and max/min-tree records
    reference/      readable construction of the RAG and component tree
    higra/          Higra-based construction and conversion to node parents
  generation/       synthetic images, noise models, seeds and correction targets
  data/             clean image sources, stored families and PyG graphs
  model.py          recurrent network with its parent and action readouts
  attention.py      graph-local parent attention, dense and FlexAttention
  losses.py         parent and action losses averaged per image
  metrics.py        accuracies and baselines per image
  train.py          epoch loop, checkpoint selection and test evaluation
  correction/       decoding predictions into the ε-tree and scoring it
scripts/            dataset generation and training entry points
notebooks/          Colab notebook that generates a dataset and trains
tests/              tests mirroring the package layout
```

Node parents are the representation every tree uses here, one parent per RAG
node with the root as its own parent. The clean component tree, the observed
tree and the model's corrective parents are all stored this way.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

To write a source of clean square images, corrupt it with salt-and-pepper
noise and train on the result:

```bash
uv run python scripts/make_squares.py source
uv run python scripts/make_dataset.py source dataset --noise salt_and_pepper
uv run python scripts/train.py dataset run/main --max-epochs 100
```

Training uses a CUDA GPU when one is available, then Apple's MPS backend, then
the CPU. The Colab notebook runs the same steps on a hosted GPU and keeps
datasets and runs in a `letree` folder of Google Drive.

## License

MIT. See [LICENSE](LICENSE).
