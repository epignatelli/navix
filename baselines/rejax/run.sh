# Algorithms and config files
ALGOS="ppo sac dqn pqn iqn"
CONFIGS=(./configs/*.yaml)
MAX_JOBS=4 # Max number of parallel jobs

# if MAX_JOBS > 1 then add openblas env vars
if [[ $MAX_JOBS -gt 1 ]]; then
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=$(1/($MAX_JOBS + 1))
  export JAX_LOG_COMPILES=0
fi

# Ensure log directory exists
mkdir -p logs

# Function to block if too many jobs are running
wait_for_slot() {
  while (($(jobs -r | wc -l) >= MAX_JOBS)); do
    sleep 1
  done
}

start=$(date +%s)

# Loop over all config-algo pairs
for algorithm in $ALGOS; do
  for config_file in "${CONFIGS[@]}"; do
    wait_for_slot
    (
      # Unique cache dir to avoid write conflicts
      export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache_${algorithm}_$(basename "$config_file" .yaml)_$RANDOM"

      echo "Running $algorithm on $config_file"
      python train.py \
        --config "$config_file" \
        --algorithm "$algorithm" \
        >logs/${algorithm}_$(basename "$config_file" .yaml).log 2>&1
    ) &
  done
done

# Wait for all jobs to complete
wait

end=$(date +%s)
echo Training completed in $(expr $end - $start) seconds.

# Plot results
echo "Plotting results..."
python plot.py
