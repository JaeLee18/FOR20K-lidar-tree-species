# Tree Species for Android (offline)

An Android app that identifies the species of a single-tree LiDAR scan **on the phone**, with no network
connection, and keeps a record of every scan: prediction, field notes, verified species, and GPS position.

![Records, a prediction with its 3D view and top five species, the field record with verified species, notes and GPS position, and dark mode](../docs/assets/android_app.jpg)

- **Input:** `.las`, `.laz`, `.ply`, `.pcd`, `.xyz`, `.txt`, `.csv`, `.pts`, `.npy`, `.npz`. Pick files with
  **Add scans**, or **Share** / **Open with** from a file manager or scanner app.
- **Output:**
  - the most likely of the 33 FOR-species20K species, and the top five with probabilities
  - an interactive 3D view of the points
  - the seven renders the model saw
- **Records:**
  - verified species, site or plot, notes, and the phone's GPS position
  - search, and an agreement rate between the model and your field identifications
  - CSV export with the same columns as the [web app](../docs/web_app.md), plus latitude and longitude
- **Requirements:** Android 8.0+ (API 26), 64-bit (arm64-v8a). The app is about 330 MB, and the model takes
  about 1 GB of RAM while running.

## The phone model

The full model, five folds at two resolutions (5.3 GB), does not fit on a phone. The app runs **one fold
(fold 0) of the 256 px ConvNeXtV2-Base ×3 network**, without test-time augmentation. Its weights are stored as
8-bit integers (dynamic per-channel quantisation of every MatMul), which takes it from 1,016 MB to 294 MB.

| Fold 0's 3,542 held-out trees (natural class mix, no TTA) | OA | Macro-F1 |
|---|---|---|
| Same network in PyTorch, fp32 | 0.8905 | 0.8488 |
| **Phone model (ONNX int8, 294 MB)** | **0.8907** | **0.8493** |

Quantisation costs nothing measurable: both models give the same top species for 98.9% of trees. For
reference, the full desktop model (both resolutions, five folds each, plus test-time augmentation) scores
0.899 OA out-of-fold and 85.0% on the official, species-balanced hidden test. The numbers are stored in
`app/src/main/assets/model_info.json`, and the app shows them under **About the model**.

**Speed.** On an x86_64 emulator with 8 cores, a tree takes about 4 s on the CPU. Rendering takes under 0.2 s, and the
three ConvNeXt networks take the rest. It has not yet been timed on a physical phone.

## Build

```bash
# 1. export the model into the app (needs the released 256 px weights in ./weights; ~30 min with the accuracy check)
python scripts/export/export_mobile.py --img-size 256 --fold 0 --per-channel --out android/app/src/main/assets \
    --eval-renders data/renders_256/dev        # optional: renders of the public set for the accuracy check

# 2. build (JDK 17 and the Android SDK; Android Studio works too)
cd android
./gradlew :core:test            # Kotlin port vs. the Python package (below)
./gradlew :app:assembleRelease  # app/build/outputs/apk/release/app-release.apk
adb install app/build/outputs/apk/release/app-release.apk
```

The release build is signed with the debug key so it installs directly. To publish it, sign it with your own
keystore. A Play Store build would also have to move the 294 MB model into Play Asset Delivery.

## How the port is checked

`core/` is plain Kotlin with no Android code: point-cloud readers, the 7-view renderer, input normalisation, and
the ONNX classifier. That lets `./gradlew :core:test` run on any machine against golden data that
`tools/make_golden.py` writes from the Python package:

- **Readers:** the same synthetic tree in all eleven formats and versions must give byte-identical float32
  points to `mvtree.io`.
- **Renderer:** the four example trees, 7 views × 256² pixels each, must match `mvtree.render` pixel for pixel.
  The released weights only work on renders made exactly this way.
- **Model:** the full Kotlin pipeline's probabilities must match Python ONNX Runtime on the same file.

## Layout

```
android/
├── core/        # Kotlin/JVM: PointClouds (readers), Renderer, Classifier (ONNX Runtime), Viz; host unit tests
├── app/         # Android UI (Jetpack Compose): records list, record screen, 3D viewer, CSV export
└── tools/make_golden.py   # golden test data from the Python package
```

Dependencies: ONNX Runtime (MIT), laszip4j (LGPL-2.1, LAZ decoding), AndroidX / Jetpack Compose (Apache-2.0).
The app is GPL-3.0, like the rest of the repository.
