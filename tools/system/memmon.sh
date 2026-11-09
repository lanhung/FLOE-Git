#!/bin/bash
# 容器内存采样（cgroup v2）。用法: bash tools/system/memmon.sh <输出csv> [采样间隔秒]
#
# 验证路径一个样本就是一整景（最大约 8GB），多 worker 时经 /dev/shm 传递，记在 shmem 而非
# anon；tmpfs 页不可回收，所以调 num_workers_val 时要同时看 shmem 与 current。
#
# 输出列: t（epoch 秒）,anon,file,shmem,current,shm_used_kb
OUT="${1:?用法: memmon.sh <输出csv> [间隔秒]}"
INT="${2:-2}"
echo "t,anon,file,shmem,current,shm_used_kb" > "$OUT"
while true; do
  A=$(grep '^anon '  /sys/fs/cgroup/memory.stat | awk '{print $2}')
  F=$(grep '^file '  /sys/fs/cgroup/memory.stat | awk '{print $2}')
  S=$(grep '^shmem ' /sys/fs/cgroup/memory.stat | awk '{print $2}')
  C=$(cat /sys/fs/cgroup/memory.current)
  D=$(df /dev/shm | tail -1 | awk '{print $3}')
  echo "$(date +%s),$A,$F,$S,$C,$D" >> "$OUT"
  sleep "$INT"
done
