#!/usr/bin/env bash
# Keep a long OWL fine-tuning run alive across crashes and reboots.
#
# animaloc checkpoints once per epoch, so an interruption costs at most the epoch in
# flight; `main.py --resume` then picks up from latest_model.pth at the following epoch
# with the optimizer state and best-so-far score restored.  This wrapper only automates
# that: it refuses to start a second trainer for the same run, and restarts one that dies
# until the epoch cap is reached.
#
# Run it by hand:
#   megadetector_overhead/keepalive_resume.sh &
#
# Survive reboots (user crontab -- no sudo, unlike a systemd unit, which would also need
# `sudo loginctl enable-linger zygfd` to start before login):
#   (crontab -l 2>/dev/null; echo "@reboot /home/zygfd/waterfowl/detection/megadetector_overhead/keepalive_resume.sh") | crontab -
#
# Stop it:  rm -f <run_dir>/keepalive.enabled   (takes effect at the next restart), or
#           pkill -f keepalive_resume.sh && pkill -f tools/train.py
set -uo pipefail

REPO="${REPO:-/home/zygfd/waterfowl/detection}"
PY="${PY:-/home/zygfd/anaconda3/envs/waterfowl/bin/python}"
MODEL="${MODEL:-OWLD_H}"
CKPT_SUBDIR="${CKPT_SUBDIR:-mdo_owl_d}"
RUN_TS="${RUN_TS:-2026-09-08_12-57-04}"
EPOCHS="${EPOCHS:-100}"
MAX_FAILS="${MAX_FAILS:-5}"     # consecutive *fast* failures before giving up
FAST_FAIL_S="${FAST_FAIL_S:-600}"  # a leg shorter than this counts as a fast failure
BACKOFF="${BACKOFF:-120}"       # seconds to wait before retrying

RUN_DIR="$REPO/output/checkpoints/$CKPT_SUBDIR/$RUN_TS"
LOG="$RUN_DIR/keepalive.log"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 1; }

# Single instance, held for the life of the script.  Without this a @reboot line and a
# hand-started copy would race and put two trainers on one GPU.
exec 9>"$RUN_DIR/keepalive.lock"
flock -n 9 || { log "another keepalive holds the lock; exiting"; exit 0; }

# A kill switch that survives reboots, so the run can be retired without editing crontab.
touch "$RUN_DIR/keepalive.enabled"

# The epoch recorded in the checkpoint; mmap'd so this does not pull 3.5 GB into RAM.
done_epoch() {
  "$PY" - "$RUN_DIR/latest_model.pth" <<'EOF' 2>/dev/null || echo -1
import sys
try:
    import torch
    print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True)["epoch"]))
except Exception:
    print(-1)
EOF
}

fails=0
while :; do
  [ -e "$RUN_DIR/keepalive.enabled" ] || { log "keepalive.enabled removed; stopping"; exit 0; }

  # Never start a second trainer for this run -- the trainer's cmdline carries the run
  # timestamp via --config-dir, which is what makes this check specific to *this* run.
  if pgrep -f "tools/train.py .*$RUN_TS" >/dev/null 2>&1; then
    log "trainer for $RUN_TS already running; nothing to do"
    exit 0
  fi

  ep="$(done_epoch)"
  if [ "$ep" -lt 0 ]; then
    log "FATAL: cannot read an epoch from $RUN_DIR/latest_model.pth; refusing to start"
    log "       (resuming blind would restart from epoch 1 and overwrite the run)"
    exit 1
  fi
  if [ "$ep" -ge "$EPOCHS" ]; then
    log "epoch $ep >= cap $EPOCHS -- training complete, nothing to resume"
    exit 0
  fi

  leg_log="$RUN_DIR/resume_$(date +%Y%m%d_%H%M%S)_launch.log"
  log "resuming at epoch $((ep + 1)) of $EPOCHS -> $leg_log"
  start=$SECONDS
  "$PY" "$REPO/megadetector_overhead/main.py" \
      --train --model "$MODEL" --resume "$RUN_TS" --epochs "$EPOCHS" \
      > "$leg_log" 2>&1 < /dev/null
  rc=$?
  elapsed=$((SECONDS - start))

  if [ $rc -eq 0 ]; then
    log "trainer exited cleanly after ${elapsed}s"
    exit 0
  fi

  # A leg that ran for a while then died is a crash worth retrying; one that dies
  # immediately is usually a config or environment error that retrying will not fix.
  if [ $elapsed -lt $FAST_FAIL_S ]; then
    fails=$((fails + 1))
    log "exit $rc after only ${elapsed}s (fast failure $fails/$MAX_FAILS); see $leg_log"
    if [ $fails -ge $MAX_FAILS ]; then
      log "giving up after $MAX_FAILS fast failures -- needs a human"
      exit 1
    fi
  else
    fails=0
    log "exit $rc after ${elapsed}s; retrying"
  fi
  sleep "$BACKOFF"
done
