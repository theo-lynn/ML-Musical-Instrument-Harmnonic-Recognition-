# Musical Instrument Recognition from Harmonics

Identifying a musical instrument from a short audio clip by looking at the frequency spectrum of its harmonics.

Every instrument produces a fundamental note plus a characteristic set of overtones (its *timbre*). A **mel spectrogram** turns those harmonics into an image — frequency on the y-axis, time on the x-axis — so the problem becomes image classification. A pretrained ResNet34 is fine-tuned on the spectrograms with [fastai](https://docs.fast.ai/).

<a href="https://colab.research.google.com/github/theo-lynn/ML-Musical-Instrument-Harmnonic-Recognition-/blob/main/instrument_classifier.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Dataset

[Musical Instruments Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset) on Kaggle (CC0) — guitar, drums, violin and piano clips, with separate train and test splits.

## Pipeline

1. **Download** the dataset with the Kaggle API
2. **Clean** — hash every clip to remove exact duplicates, clips filed under two different instruments, and test clips that also appear in train
3. **Spectrograms** — trim/pad every clip to 3 s and convert it to a 128-band mel spectrogram (dB), saved as a grayscale image
4. **Train** — fine-tune ResNet34 (no flip/rotate augmentations, since flipping a spectrogram swaps high and low frequencies)
5. **Evaluate** — confusion matrix, top losses, and a per-class report on the held-out test set
6. **Export** the model and predict on any audio file with `predict_instrument(path)`

## Running it

**Colab (recommended):** open the notebook with the badge above, switch to a GPU runtime, and add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets (from *kaggle.com → Settings → API → Create New Token*). Then *Runtime → Run all*.

**Locally:**

```bash
pip install -r requirements.txt
# put kaggle.json in ~/.kaggle/
jupyter notebook instrument_classifier.ipynb
```

## Results

The first version of this project (3 classes, no data cleaning, variable-length spectrograms rendered through matplotlib) reached around **60% validation accuracy** (≈40% error rate). The pipeline has since been rebuilt as described above — re-run the notebook to get updated numbers.
