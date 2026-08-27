#!/bin/bash
# Sample GPU util/mem/clock every 5s to <out> while a job runs (pid $1).
OUT="$2"
PID="$1"
echo "ts util% memMB" > "$OUT"
while kill -0 "$PID" 2>/dev/null; do
  echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')" >> "$OUT"
  sleep 5
done
echo "sampling done -> $OUT"