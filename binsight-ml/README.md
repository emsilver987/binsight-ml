# binsight-ml

One training pipeline, two configs. Run commands from this folder.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## TrashNet baseline

Images live in `data/trashnet/dataset-resized`. Rebuild the manifest if that folder changes:

```bash
.venv/bin/python src/make_manifest_trashnet.py --config configs/trashnet.yaml
```

Train and save weights, metrics, and the confusion matrices to `runs/trashnet/`:

```bash
.venv/bin/python src/train.py --config configs/trashnet.yaml
```

On this laptop CPU that is about 10 minutes: roughly 16 seconds per epoch, 5 frozen-head epochs, then up to 20 fine-tuning epochs. Early stopping often ends phase 2 sooner. A Colab T4 finishes the same run in a few minutes. The first launch also downloads the MobileNetV2 ImageNet weights (~2 MB).

This does not need to run before Thursday. The pipeline is already checked (`--smoke` wrote `runs/trashnet_smoke`). Thursday is for the serial-command work. Train TrashNet when you want the baseline number in `runs/trashnet/metrics.json` and the weights in `runs/trashnet/model.keras` for the later comparison.

Smoke test, 8 images per class, one epoch per phase:

```bash
.venv/bin/python src/train.py --config configs/trashnet.yaml --smoke
```

## BinSight

Same command, other config. Output goes to `runs/binsight/` and does not touch `runs/trashnet/`.

```bash
.venv/bin/python src/train.py --config configs/binsight.yaml
```

This needs `data/binsight/` images and `data/manifests/binsight.csv` with columns `filepath,item_id,label`. `label` is a class name from `rules.yaml`. `item_id` groups every frame of one physical item so the split keeps them together.

Re-score a finished run without training again:

```bash
.venv/bin/python src/evaluate.py --run runs/trashnet --config configs/trashnet.yaml
.venv/bin/python src/evaluate.py --run runs/binsight --config configs/binsight.yaml
```
