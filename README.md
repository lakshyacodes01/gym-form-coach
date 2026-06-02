---
title: Gym Form Coach
emoji: 🏋️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Gym Form Coach

Real-time squat **rep counter** and **form classifier** that runs in the browser. Pose estimation happens on-device; a small neural network grades each completed rep as good or bad form.

**Live demo:** https://lakshya0822-gym-form-coach.hf.space

> **Currently supports squats.** More exercises (push-ups, lunges, and others) will be added over time — the same pose → rep-counter → classifier pipeline is built to extend to them.

<!-- Add a demo GIF here. The easiest way that avoids committing a binary:
     edit this README on github.com, drag your GIF into the editor (GitHub hosts
     it on its CDN), and paste the generated URL below. -->
![demo](REPLACE_WITH_YOUR_GIF_URL)

## What it does

- Tracks your body with on-device pose estimation (33 landmarks at ~30 fps).
- Counts squat reps with a joint-angle state machine that works from the front and the side, and ignores non-squats like walking or single-leg lifts.
- Sends each finished rep's landmarks to a backend model that returns a **good / bad** form verdict with a confidence score.

## How it works

Pose estimation and rep counting run **entirely in the browser**. Only small per-rep arrays of landmark coordinates — never video frames — are sent to the server, which keeps the experience low-latency and avoids uploading any imagery.

```
Browser:  webcam -> MediaPipe pose -> rep counter (rule-based)
                                          |
                              per-rep landmarks (small JSON)
                                          v
Server:   FastAPI -> 1D-CNN form classifier -> { "label": "good", "prob_good": 0.93 }
```

A deliberate design choice: only the **form judgment** is machine learning. Rep *counting* is rule-based geometry (a knee-angle state machine with hysteresis plus a hip-drop check), because counting is a deterministic problem and a rule is the right tool. The model is reserved for the genuinely fuzzy task — judging form quality.

- **Pose:** MediaPipe Pose Landmarker (pretrained), run client-side via WebAssembly.
- **Rep counting:** knee-angle state machine + hip-drop gate; picks the more visible leg so it works from any angle.
- **Form classifier:** a 1D-CNN trained in PyTorch on normalized landmark sequences — each rep is recentered on the hips, scaled by torso length (so it's invariant to position and camera distance), and resampled to a fixed 32 frames.

## Results & honest evaluation

Trained on 133 self-recorded squat reps labeled good or bad.

On a random train/test split the model scored **100%** — which I treated as a red flag, not a success. Consecutive reps in a session are near-identical, so a random split leaks near-duplicate reps across train and test, and the model effectively recognizes reps it has already seen.

Evaluating instead on a **completely separate recording session** (verified to have zero overlap with the training data) gives the honest figure:

| Metric | Value |
|---|---|
| Accuracy (held-out session) | **0.87** |
| Precision / recall (good) | 0.82 / 0.96 |
| Precision / recall (bad) | 0.95 / 0.78 |

**Caveats, stated plainly:** this is a single subject from one camera angle, so accuracy on a new person would be lower. The model also leans slightly toward predicting "good" — it misses about 22% of bad reps — which for a form coach is the error type worth improving first.

## Tech stack

- **Frontend:** React, Vite, MediaPipe Tasks (Web)
- **Backend:** FastAPI, PyTorch (CPU inference)
- **Infra:** Docker, GitHub Actions (CI/CD), Hugging Face Spaces

## Run locally

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt torch
uvicorn app.main:app --port 7860

# Frontend (in a second terminal)
cd frontend
npm install
npm run dev
```

Open the Vite URL, allow the camera, and do a squat — the rep count ticks up and the form verdict appears after you stand back up.

## Project structure

```
backend/      FastAPI app, the trained model (as base64 text), and tests
frontend/     React + Vite client (pose tracking, rep counter, UI)
.github/      CI/CD workflow (lint, test, build, deploy)
Dockerfile    Multi-stage build: compiles the frontend, serves it from FastAPI
```

## Limitations & future work

- **More exercises:** squats only for now; push-ups, lunges, and others will be added over time, each as its own classifier on the same landmark pipeline.
- **Generalization:** trained on one person; adding multiple subjects and camera angles would give a real cross-person accuracy number.
- **Granularity:** currently binary good/bad; a natural extension is multi-label fault detection (shallow depth, knees caving, forward lean).
- **Progress tracking:** persisting workout sessions to a database with a trends dashboard is planned.

## Acknowledgements

Pose estimation by [MediaPipe](https://ai.google.dev/edge/mediapipe).
