vllm serve Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --max-num-seqs 1024