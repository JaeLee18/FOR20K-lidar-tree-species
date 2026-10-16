#!/usr/bin/env bash
# Watch a run started by tpu/cloud/launch.sh: print progress, and recreate the VM when a spot TPU is
# preempted, trying each "TYPE ZONE VERSION" line of CANDIDATES in turn until one has capacity. The VM is
# looked up in those zones and in ZONE. Exits when the VM reports DONE (0) or FAILED (1). Takes the same
# environment variables as launch.sh.
#
#   BUCKET=gs://my-bucket bash tpu/cloud/watch.sh
#   BUCKET=gs://my-bucket CANDIDATES=$'v6e-4 us-east1-d v2-alpha-tpuv6e\nv5p-8 us-east5-a v2-alpha-tpuv5' \
#     bash tpu/cloud/watch.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET=${BUCKET:?set BUCKET=gs://...}
CONFIG=${CONFIG:-configs/convnextv2_256.yaml}
RUN=${RUN:-tpu_$(basename "$CONFIG" .yaml)}
NAME=${NAME:-three-lidars-${RUN//_/-}}
CANDIDATES=${CANDIDATES:-"${TYPE:-v6e-4} ${ZONE:-us-central1-b} ${VERSION:-v2-alpha-tpuv6e}"}
INTERVAL=${INTERVAL:-300}
mapfile -t CANDS <<< "$CANDIDATES"
c=0

lookup() {  # sets state and zone of the VM, searched in every candidate zone; fails if a zone cannot be listed
  local z s
  state="" zone=""
  for z in $(printf '%s\n' "${ZONE:-}" "${CANDS[@]#* }" | awk 'NF {print $1}' | sort -u); do
    s=$(gcloud compute tpus tpu-vm list --zone "$z" --filter "name~/$NAME\$" --format 'value(state)' 2>/dev/null) || return 1
    if [[ -n $s ]]; then state=$s zone=$z; fi
  done
}

while true; do
  for marker in DONE FAILED; do
    if gcloud storage ls "$BUCKET/$RUN/$marker" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] $marker"
      gcloud storage cat "$BUCKET/$RUN/logs/vm.log" 2>/dev/null | grep -E "fold|OA|F1|done|Error" | tail -15
      [[ $marker == DONE ]] && exit 0 || exit 1
    fi
  done
  if ! lookup; then  # an API error is not a missing VM: recreating it then would start a second copy
    echo "[$(date '+%F %T')] $NAME: lookup failed, trying again later"
    sleep "$INTERVAL"
    continue
  fi
  echo "[$(date '+%F %T')] $NAME: ${state:-absent} ${zone}"
  case "${state:-}" in
    PREEMPTED|TERMINATED|STOPPED|"")
      [[ -n "${state:-}" ]] && gcloud compute tpus tpu-vm delete "$NAME" --zone "$zone" --quiet
      for _ in "${CANDS[@]}"; do
        read -r type zone version <<< "${CANDS[$c]}"
        if TYPE=$type ZONE=$zone VERSION=$version UPLOAD_CODE=0 bash "$HERE/launch.sh" >/dev/null 2>&1; then
          echo "[$(date '+%F %T')] recreated as $type in $zone"
          break
        fi
        echo "[$(date '+%F %T')] no $type in $zone"
        c=$(( (c + 1) % ${#CANDS[@]} ))
      done ;;
    *)
      gcloud storage cat "$BUCKET/$RUN/logs/vm.log" 2>/dev/null | grep -E "epoch=|step|fold [0-9]|trees/s" | tail -2 ;;
  esac
  sleep "$INTERVAL"
done
