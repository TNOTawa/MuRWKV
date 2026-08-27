#!/usr/bin/env bash
# Quick status of the data-disk downloads (no GPU needed).
echo "== Slakh2100-16k tar (choihy, 307 GB) =="
S=$(du -sb /root/autodl-tmp/data/slakh_yourmt3_16k/*.p* 2>/dev/null | awk '{s+=$1} END {printf "%.2f", s/1e9}')
N=$(ls /root/autodl-tmp/data/slakh_yourmt3_16k/*.p* 2>/dev/null | wc -l)
echo "downloaded: ${S} GB over ${N} parts (target 307.17 GB)"
pgrep -f fetch_parallel.py > /dev/null && echo "download process: RUNNING" || echo "download process: DEAD"
tail -2 /root/autodl-tmp/data/slakh_yourmt3_16k/fetch.log 2>/dev/null
echo "== 16k corpus (resampled from FLAC-redux) =="
find /root/autodl-tmp/data/slakh2100_16k_from_flac -name mix.wav 2>/dev/null | wc -l
echo "tracks (target 1709)"
echo "== disk =="
df -h /root/autodl-tmp | tail -1