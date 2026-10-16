#!/usr/bin/env bash
# Create a Cloud TPU VM that trains all five folds of one config, with everything kept in a GCS bucket.
#
#   BUCKET=gs://my-bucket bash tpu/cloud/launch.sh                  # v6e-4 spot in us-central1-b, 256 px
#   BUCKET=gs://my-bucket ZONE=us-east5-b TYPE=v6e-8 CONFIG=configs/convnextv2_512.yaml bash tpu/cloud/launch.sh
#
# The VM runs tpu/cloud/startup.sh at boot: it installs the environment, checks inference against the
# released weights if they are in $BUCKET/weights/, restores the renders from the bucket (or downloads
# FOR-species20K and renders it, once), resumes the run from the bucket, syncs it back every 5 minutes,
# and deletes itself when it is done (or an hour after it fails). A preempted spot VM is recreated by
# tpu/cloud/watch.sh.
#
# Bucket layout:  $BUCKET/data/renders_<size>_dev.tar    render cache shared by runs
#                 $BUCKET/weights/convnextv2_{256,512}/   released weights (optional, for the inference check)
#                 $BUCKET/$RUN/{code.tar.gz,runs/,logs/,inference/,weights_tpu/,DONE|FAILED}
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUCKET=${BUCKET:?set BUCKET=gs://...}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-us-central1-b}
TYPE=${TYPE:-v6e-4}
VERSION=${VERSION:-v2-alpha-tpuv6e}      # v5e: v2-alpha-tpuv5-lite; v5p: v2-alpha-tpuv5; v4: tpu-ubuntu2204-base
CONFIG=${CONFIG:-configs/convnextv2_256.yaml}
RUN=${RUN:-tpu_$(basename "$CONFIG" .yaml)}
NAME=${NAME:-three-lidars-${RUN//_/-}}
FIRST_FOLD=${FIRST_FOLD:-0}
LAST_FOLD=${LAST_FOLD:-4}
SPOT=${SPOT:-1}                          # 0 = on-demand
AUTO_DELETE=${AUTO_DELETE:-1}            # delete the VM when the run ends
UPLOAD_CODE=${UPLOAD_CODE:-1}            # 0 = keep the code already in the bucket (recreating a VM)
INFER_CHECK=${INFER_CHECK:-1}            # 0 = skip the inference check against the released weights
TRAIN_ARGS=${TRAIN_ARGS:-}               # extra train_tpu.py arguments, e.g. "--per-chip-batch 4" (no commas)

if [[ "$UPLOAD_CODE" == 1 ]]; then
  # The working tree as git sees it (tracked and new files, minus ignored ones), without the app and media.
  TAR=$(mktemp --suffix .tar.gz)
  (cd "$REPO" && git ls-files -co --exclude-standard | grep -v -E '^(android|docs/assets)/' \
    | tar czf "$TAR" -T -)
  gcloud storage cp "$TAR" "$BUCKET/$RUN/code.tar.gz"
  rm -f "$TAR"
fi

SPOT_FLAG=()
if [[ "$SPOT" == 1 ]]; then SPOT_FLAG=(--spot); fi
META="bucket=$BUCKET,run=$RUN,config=$CONFIG,first_fold=$FIRST_FOLD,last_fold=$LAST_FOLD,name=$NAME,zone=$ZONE"
META+=",auto_delete=$AUTO_DELETE,infer_check=$INFER_CHECK${TRAIN_ARGS:+,train_args=$TRAIN_ARGS}"
gcloud compute tpus tpu-vm create "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --accelerator-type "$TYPE" --version "$VERSION" "${SPOT_FLAG[@]}" \
  --scopes https://www.googleapis.com/auth/cloud-platform --labels "app=three-lidars,run=${RUN//_/-}" \
  --metadata "$META" \
  --metadata-from-file "startup-script=$REPO/tpu/cloud/startup.sh"

echo "created $NAME ($TYPE, $ZONE). Log: gcloud storage cat $BUCKET/$RUN/logs/vm.log | tail"
