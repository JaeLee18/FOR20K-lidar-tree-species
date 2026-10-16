#!/bin/bash
# Boot script of the training TPU VM (installed by tpu/cloud/launch.sh; runs as root at every boot).
# Every stage is resumable, so a recreated VM continues from what the bucket holds.
exec >> /var/log/three-lidars.log 2>&1
set -x
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
attr() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1"; }
BUCKET=$(attr bucket)
RUN=$(attr run)
CONFIG=$(attr config)
FIRST=$(attr first_fold || echo 0)
LAST=$(attr last_fold || echo 4)
NAME=$(attr name)
ZONE=$(attr zone)
AUTO_DELETE=$(attr auto_delete || echo 0)
INFER_CHECK=$(attr infer_check || echo 1)
TRAIN_ARGS=$(attr train_args || true)
W=/opt/three-lidars
REPO=$W/repo
PY=$W/venv/bin/python
OUT=runs/$RUN
export HOME=/root PJRT_DEVICE=TPU

up_log() { gcloud storage cp -q /var/log/three-lidars.log "$BUCKET/$RUN/logs/vm.log"; }
sync_up() {
  gcloud storage rsync -q "$REPO/$OUT" "$BUCKET/$RUN/runs" --exclude '.*\.tmp$'
  up_log
}
finish() {  # $1 = DONE or FAILED
  sync_up
  echo "$1 $(date -u +%FT%TZ)" | gcloud storage cp - "$BUCKET/$RUN/$1"
  if [ "$1" = FAILED ]; then  # an hour to look at a failed VM; `touch /opt/three-lidars/KEEP` keeps it
    sleep 3600
    [ -f "$W/KEEP" ] && exit 0
  fi
  if [ "$AUTO_DELETE" = 1 ]; then
    gcloud compute tpus tpu-vm delete "$NAME" --zone "$ZONE" --quiet --async
  fi
  exit 0
}
if gcloud storage ls "$BUCKET/$RUN/DONE" >/dev/null 2>&1; then finish DONE; fi
gcloud storage rm -q "$BUCKET/$RUN/FAILED" 2>/dev/null  # this boot tries again

# 1. code (refreshed at every boot)
mkdir -p "$REPO" && cd "$W" || finish FAILED
gcloud storage cp "$BUCKET/$RUN/code.tar.gz" . && tar xzf code.tar.gz -C "$REPO" || finish FAILED
cd "$REPO" || finish FAILED

# 2. environment
if [ ! -f "$W/.env_ready" ]; then
  for i in $(seq 10); do  # apt may be locked by unattended-upgrades right after boot
    dpkg -s python3-venv >/dev/null 2>&1 && break
    apt-get update -qq && apt-get install -y -qq python3-venv && break
    sleep 30
  done
  VENV=$W/venv bash tpu/setup_vm.sh && touch "$W/.env_ready" || finish FAILED
fi
up_log

# 3. inference check with the released weights (once per run, if they are in the bucket)
if [ "$INFER_CHECK" = 1 ] && gcloud storage ls "$BUCKET/weights/convnextv2_256/config.json" >/dev/null 2>&1 \
   && ! gcloud storage ls "$BUCKET/$RUN/inference/DONE" >/dev/null 2>&1; then
  mkdir -p weights inference && gcloud storage rsync -r "$BUCKET/weights" weights
  infer() {  # $1 = output name, then predict_tpu.py arguments
    local name=$1; shift
    $PY tpu/predict_tpu.py "$@" -q 2> "inference/$name.log"
    echo "$name exit $?" >> inference/STATUS
  }
  # the four example scans, then 64 copies of them for throughput
  infer examples_bf16 examples/ -o inference/examples_bf16.json
  infer examples_fp32 examples/ -o inference/examples_fp32.json --precision fp32
  mkdir -p /tmp/bench && for c in $(seq -w 16); do
    for f in examples/*.laz; do ln -sf "$REPO/$f" "/tmp/bench/${c}_$(basename "$f")"; done
  done
  infer bench64 /tmp/bench -o inference/bench64.csv
  infer bench64_1chip /tmp/bench -o inference/bench64_1chip.csv --chips 1
  cp inference/STATUS inference/DONE
  gcloud storage rsync -r inference "$BUCKET/$RUN/inference"
fi
up_log

# 4. renders: from the bucket's cache, or download FOR-species20K and render it once
SIZE=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['img_size'])" "$CONFIG")
CACHE=$BUCKET/data/renders_${SIZE}_dev.tar
if [ ! -f "$W/.data_ready" ]; then
  mkdir -p data
  if gcloud storage ls "$CACHE" >/dev/null 2>&1; then
    gcloud storage cp "$CACHE" - | tar xf - -C data || finish FAILED
  else
    $PY scripts/data/download_data.py --only dev || finish FAILED
    $PY scripts/data/render_dataset.py --source data/raw/dev.zip --out "data/renders_$SIZE/dev" --size "$SIZE" \
      --workers "$(nproc)" || finish FAILED
    tar cf - -C data "renders_$SIZE/dev" | gcloud storage cp - "$CACHE"
    rm -f data/raw/dev.zip
  fi
  touch "$W/.data_ready"
fi
up_log

# 5. train, resuming from the bucket and syncing back every 5 minutes
mkdir -p "$OUT"
if gcloud storage ls "$BUCKET/$RUN/runs/" >/dev/null 2>&1; then
  gcloud storage rsync -q "$BUCKET/$RUN/runs" "$OUT"
fi
( while sleep 300; do sync_up; done ) &
SYNC=$!
# A crash of the TPU runtime costs only the epoch in progress: train again from the last checkpoint.
for ATTEMPT in 1 2 3; do
  PY=$PY TRAIN_ARGS=$TRAIN_ARGS bash tpu/train_all_folds_tpu.sh "$CONFIG" "$OUT" "$FIRST" "$LAST"
  STATUS=$?
  [ $STATUS -eq 0 ] && break
  echo "training exited with status $STATUS (attempt $ATTEMPT)"
  pkill -9 -f '[s]pawn_main'  # chip processes left behind would hold the TPU
  sleep 60
done
kill $SYNC
if [ $STATUS -ne 0 ]; then finish FAILED; fi

# 6. predict the examples with the TPU-trained weights, and keep the weights
$PY tpu/predict_tpu.py examples/ --weights weights_tpu -o "$OUT/examples_tpu_weights.json" 2>&1 | tail -30
gcloud storage rsync -r weights_tpu "$BUCKET/$RUN/weights_tpu"
finish DONE
