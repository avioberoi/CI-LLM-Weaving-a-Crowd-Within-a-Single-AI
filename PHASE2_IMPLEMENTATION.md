# Phase 2 Implementation: DeepSpeed Integration and Parallel Agent Processing

## Overview

Phase 2 builds upon the Phase 1 Accelerate integration to add:
1. **DeepSpeed ZeRO-2** optimization for memory-efficient training of larger models
2. **Parallel agent forward pass** implementation to reduce training time
3. **Detailed profiling** capabilities for performance analysis
4. **Multi-node training preparation** through proper DeepSpeed/Accelerate configuration

## Key Features Implemented

### 1. DeepSpeed Integration

#### Configuration
- **Programmatic Configuration**: Automatic DeepSpeed setup when `use_deepspeed: true` in configs.yaml
- **File-based Configuration**: Support for custom DeepSpeed JSON config via `deepspeed_config_file` parameter
- **ZeRO-2 Optimization**: Default configuration uses ZeRO Stage 2 for optimizer and gradient sharding

#### Usage
```yaml
# In configs.yaml:
use_deepspeed: true
deepspeed_config_file: "ds_config_zero2.json"  # Optional, uses programmatic config if not specified
```

Launch training with DeepSpeed:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

#### Memory Benefits
- Significantly reduced per-GPU memory footprint
- Enables training of Gemma-2-9B with K=4+ agents on 40GB A100s
- Optimizer states and gradients are sharded across GPUs

### 2. Parallel Agent Forward Pass

#### Architecture
The parallel agent implementation creates K separate `PeftModel` instances that share the same frozen base model weights. This avoids the overhead of adapter switching while maintaining memory efficiency.

#### Implementation Details
- **Shared Base Model**: All K PeftModel instances reference the same base model object
- **Memory Efficiency**: PyTorch shares the underlying parameter storage for the frozen base weights
- **DropConnect Support**: Properly handles weight masking/restoration for each parallel model

#### Configuration
```yaml
# In configs.yaml:
parallel_mode: "parallel"  # Options: "sequential" (original), "parallel" (new)
```

#### Performance Considerations
- Reduces adapter switching overhead
- Currently executes sequentially on same GPU (true parallelism requires multi-GPU placement)
- Future enhancement: Distribute agents across GPUs for true parallel execution

### 3. Detailed Profiling

#### Features
- **Automatic Timing**: Tracks time for each training component
- **Memory Monitoring**: Logs GPU memory usage at key points
- **Configurable**: Enable with `profile_steps` parameter

#### Profiled Components
- Data transfer to GPU
- Forward pass (total and per-agent if feasible)
- Loss calculation (task loss and SWD)
- Backward pass
- Optimizer step

#### Usage
```yaml
# In configs.yaml:
profile_steps: 5  # Profile first 5 steps
log_memory_usage: true
```

#### Output Example
```
--- Performance Summary at Step 5 ---
data_transfer: 0.0234s (avg over 5 calls)
forward_pass: 1.2456s (avg over 5 calls)
loss_calculation: 0.0567s (avg over 5 calls)
backward_pass: 0.8901s (avg over 5 calls)
optimizer_step: 0.1234s (avg over 5 calls)

[Step 5] GPU Memory - Allocated: 28.45GB, Max Allocated: 32.10GB, Reserved: 35.00GB
```

### 4. Enhanced Checkpointing

- **Parallel Mode Support**: Properly saves/loads weights for parallel agent models
- **Mode Persistence**: Saves `parallel_mode` in checkpoint metadata
- **Backward Compatibility**: Can load sequential checkpoints into parallel mode

## Testing

### Dry Run Sanity Check
The updated `dry_run_sanity_check.py` includes:
- DeepSpeed configuration validation
- Parallel agent mode testing
- Memory usage verification
- Performance comparison between sequential and parallel modes

Run the comprehensive test:
```bash
python dry_run_sanity_check.py
```

### Performance Benchmarking

Example benchmark results (simulated):
| Configuration | Model | K Agents | Mode | Memory/GPU | Step Time |
|--------------|-------|----------|------|------------|-----------|
| Baseline | Gemma-2-2b | 2 | Sequential | 12GB | 0.8s |
| Baseline | Gemma-2-2b | 2 | Parallel | 12GB | 0.7s |
| DeepSpeed | Gemma-2-9b | 4 | Sequential | 35GB | 3.2s |
| DeepSpeed | Gemma-2-9b | 4 | Parallel | 35GB | 2.8s |

## Multi-Node Training Preparation

The current implementation is ready for multi-node expansion:

1. **DeepSpeed Integration**: Handles distributed optimizer/gradient management
2. **Accelerate Compatibility**: Uses Accelerate's distributed abstractions
3. **Checkpoint Management**: Distributed-aware saving/loading

### Future Multi-Node Launch
```bash
# On each node:
accelerate launch --num_machines 2 --machine_rank 0 --main_process_ip $MASTER_ADDR train.py
```

## Configuration Reference

### New Configuration Parameters

```yaml
# DeepSpeed
use_deepspeed: false          # Enable DeepSpeed optimization
deepspeed_config_file: null   # Path to custom DeepSpeed config JSON

# Parallel Agents
parallel_mode: "sequential"   # "sequential" or "parallel"

# Profiling
profile_steps: null          # Number of steps to profile (e.g., 5)
log_memory_usage: true       # Log GPU memory usage during training
```

### DeepSpeed Configuration (ds_config_zero2.json)

Key settings in the provided configuration:
- **ZeRO Stage 2**: Optimizer and gradient sharding
- **Gradient Clipping**: Set to 1.0
- **Mixed Precision**: Auto-detected (fp16/bf16)
- **Overlap Communication**: Enabled for efficiency

## Troubleshooting

### Out of Memory with DeepSpeed
- Try ZeRO Stage 3 (parameter sharding)
- Enable CPU offloading in DeepSpeed config
- Reduce batch size or sequence length

### Parallel Mode Issues
- Verify all agents have same configuration
- Check that base model weights are truly shared (monitor memory)
- Ensure checkpoint compatibility when switching modes

### Performance Degradation
- Check profiling output for bottlenecks
- Verify DeepSpeed is actually enabled (check logs)
- Consider adjusting DeepSpeed buffer sizes

## Future Enhancements

1. **True Parallel Execution**: Distribute agents across GPUs
2. **Dynamic Agent Allocation**: Adjust K based on available resources
3. **Advanced Profiling**: Integration with PyTorch Profiler for flame graphs
4. **Optimized Aggregation**: GPU-kernel for Dirichlet aggregation
5. **Multi-Node CI/CD**: Automated testing on multi-node setups 