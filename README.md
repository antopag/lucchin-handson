# Hands-on on MC simulations
### γ / hadron discrimination — Random Forest meets CNN
**INAF Lucchin PhD School · Spring 2026**

A practical companion to Antonio Pagliaro's lecture on machine learning for
high-energy astrophysics.

## Quick start

```bash
git clone <this-repo-url>
cd lucchin-2026-handson
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

If you don't yet have the ASTRI MC FITS files, you can run sections 4–6 (CNN
inference on synthetic data) without them. Section 1 needs the FITS files.

## Files

| File | What it is |
|---|---|
| `notebook.ipynb` | The hands-on notebook — work through cell by cell. |
| `synth_camera.py` | Generator of synthetic Cherenkov-like camera images (used by §4). |
| `toy_cnn.py` | Architecture of the small CNN used for inference. |
| `toy_cnn.pt` | Pre-trained CNN weights (~110 KB). |
| `requirements.txt` | Python dependencies. |
| `data/` | Place your ASTRI MC FITS files here (or set `DATA_DIR` in the notebook). |

## Naming convention for the FITS files

The notebook expects files named like:

```
MA_012_gamma_NNNNNNNN-NNNNNNNN_R_zd20-20_az000-000_train.lv2a
MA_012_proton_NNNNNNNN-NNNNNNNN_R_zd20-20_az000-000_train.lv2a
```

Drop them into `data/` (or wherever `DATA_DIR` points to) and the loader
will pick them up automatically.

## Troubleshooting

**ImportError: No module named X** — re-run `pip install -r requirements.txt`.

**Data directory not found** — edit `DATA_DIR` in section 1 of the notebook.

**FITS column name mismatch** — the loader uses placeholder column names. After
running the cell that lists the columns of one FITS file (1.2), edit
the `HILLAS_COLS` mapping in cell 1.3 to match what your FITS files actually use.

For everything else: ask in class, or come find me after.

## Data

The ASTRI Mini-Array MC FITS files are included in the `data/` folder
of this repo (~349 MB total).

— Antonio
