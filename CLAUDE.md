# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

肺影智诊 · 肺纤维化临床决策支持系统 — a Flask web app for pulmonary fibrosis CDSS. It runs a MAML-trained ResNet-18 (currently serving a two-class IPF discriminant), exposes results through a "肺纤维化分型识别" presentation layer (`PRESENTATION_CONFIG` in `pf_diagnosis_service.py`: primary diagnosis + differentials), overlays Grad-CAM, and exports a Chinese PDF report.

Environment is Windows + Python 3.12 + MySQL. Chinese is the working language for UI strings, report text, and most comments.

## Common commands

```bash
# Install deps (learn2learn only needed for training)
pip install -r requirements.txt

# One-time / after schema change: apply idempotent MySQL migrations
python db_migrations.py

# Run the app (dev)
python app.py                # binds 127.0.0.1:5000
start_server.bat             # kills anything on :5000 then launches minimized, logs to server_out.log / server_err.log

# Regenerate MAML support set (only after retraining or changing fold)
python prep_support_set.py

# Retrain MAML (needs GPU + learn2learn + full OSIC dataset under data/)
python train_model.py

# Reset a password (interactive)
python reset_password.py
python reset_patient_password.py
```

No test suite, linter, or type checker is configured — do not invent one; run the app and exercise the flow instead.

## Architecture

**Single-process Flask monolith.** `app.py` (~2300 lines) is the only route module and holds every endpoint — doctor dashboard, patient dashboard, AI diagnosis, disease management, appointments, chat, follow-up, health log, notifications, system logs. Add new routes here rather than splitting into blueprints unless explicitly asked.

**Data flow for AI diagnosis** (the paper-critical path):

1. Doctor uploads DICOM / JPG / PNG slices via `/api/ai_diagnose` → files land in `static/uploads/`.
2. `PFDianosisService` (`pf_diagnosis_service.py`) is a **global singleton** initialized once in `app.py` with `models/best_maml_fold1.pth`.
3. On each request it (a) reloads the meta-init state from `self._base_state`, (b) runs `INNER_STEPS=2` gradient steps at `INNER_LR=0.003` on the pre-baked support set at `models/ipf_support_set.pt` (2-way × 2-shot × 8 slices), (c) does per-slice inference, (d) majority-votes to the patient-level class.
4. A representative slice of the winning class is fed through `pytorch-grad-cam` (`GradCAM` on `layer4` with `eigen_smooth=True`) and saved to `static/gradcam/heatmap_<patient_id>.png`.
5. User-facing copy (UI + PDF) is generated from `PRESENTATION_CONFIG` / `DIFFERENTIAL_REFERENCE` — never leak the internal class names (相对稳定组/严重受损组, Percent≥90/≤65, "2-way 2-shot") into templates or PDFs.
6. PDF export goes through `pdf_report.py` (reportlab, A4).

**Class-label convention is load-bearing** — do not silently reorder. `CLASS_NAMES` in `pf_diagnosis_service.py`, the label logic in `osic_dataset.IPFDataset` (`Percent≤65 → 1`, `Percent≥90 → 0`, middle patients dropped), and `prep_support_set.py` all agree; if a retrained checkpoint uses the opposite ordering, swap the two entries of `CLASS_NAMES` rather than editing downstream text.

**Preprocessing must match training exactly**: lung window WC=-450, WW=1500, clipped on the raw DICOM pixel_array (no Rescale slope/intercept), resized to 224×224 bilinear, converted to RGB, then ImageNet normalize. Both `pf_diagnosis_service._load_display_image` and `osic_dataset.IPFDataset.__getitem__` implement this — keep them in sync.

**Support-set reproducibility**: `prep_support_set.py` reproduces fold 1 of a 5-fold `StratifiedKFold(shuffle=True, random_state=42)` over patients, then samples 2 patients per class with `random.Random(TEST_SEED + FOLD_INDEX)` and 8 slices per patient with `random.Random(VAL_SEED + FOLD_INDEX)`. Any change to the seeds or fold index invalidates `models/ipf_support_set.pt`.

**Database layer**: `database.Database` is a per-request `mysql-connector-python` connection wrapper — callers must `connect()` then `disconnect()` explicitly, and query methods return dict rows. There is no ORM and no connection pool. The schema is not created by the app; assume it already exists in `rare_disease_diagnosis`. `db_migrations.py` only adds columns that are missing (currently `patients.is_deleted` for the soft-delete recycle bin).

**Auth**: Flask-Login with a `users` table (`role` = `doctor` | `patient`). Doctor routes gate on `current_user.user_type == 'doctor'`. Passwords are `werkzeug.security` hashes.

**Templates**: Jinja2 under `templates/`, extending `base.html`. Static assets under `static/`; runtime-generated files (`uploads/`, `previews/`, `thumbnails/`, `gradcam/`) are gitignored.

**PDF report** (`pdf_report.py`) registers Windows system fonts from `C:\Windows\Fonts` (SimHei + Deng/DengB). On non-Windows hosts the registration will silently fall back and Chinese glyphs may render as boxes — flag this rather than swapping fonts.

## Repository conventions

- `config.py` hard-codes MySQL credentials (`root` / `981812`) and `SECRET_KEY`. Treat these as local-dev defaults; don't commit real secrets and don't rewrite the file to load from env unless asked.
- `.gitignore` excludes `data/`, all `*.pth`/`*.pt`, and `static/uploads|previews|thumbnails|gradcam/`. Do not commit checkpoints, the OSIC dataset, or user uploads.
- Only one server instance can run at a time (single MySQL DB, port 5000). Use `start_server.bat` to avoid stale processes.
- User-visible strings and report copy are Chinese; keep new strings in Chinese to match.
