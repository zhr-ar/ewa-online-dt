# Memory Leak Fixes in EWA VQ PQ

This document consolidates the two memory-leak investigations and fixes applied to the EWA VQ PQ training pipeline.

## Fix 1: Stop accumulating batch history

### Problem

Even after removing trajectory data storage, history arrays were still growing on every batch:

```python
self.attraction_history.append(iteration_data)
self.code_usage_history.append(total_code_usage)
self.reward_history.append(avg_reward)
```

With 5000+ pretrain batches plus online iterations, history grew linearly until the instance crashed.

### Solution

Store only current-batch metrics (overwrite, don't accumulate):

```python
self.current_batch_metrics = {
    'avg_attraction': avg_attraction,
    'code_usage': total_code_usage,
    'reward': avg_reward
}
```

Only online-iteration metrics (max_online_iters) need to be retained for plotting, not every pretrain batch.

## Fix 2: Stop storing full trajectory step data

### Problem

The trajectory-building loop stored full action vectors and per-step lists for every timestep:

```python
step_info = {
    'a_t': actions[t].detach().cpu().numpy(),
    'code_indices': [...],
    'subspace_attractions': [...],
    ...
}
D_trajectory.append(step_info)
```

Memory scaled as: batch_size × trajectory_length × steps → GB of accumulated data over thousands of batches.

### Solution

Compute running aggregates during the loop; return only summary metrics:

```python
step_attraction = sum(self.subspace_attractions[s][code_indices_list[s][t]].item()
                      for s in range(self.num_subspaces))
total_attraction_sum += step_attraction

D_trajectory = [{
    'avg_attraction': avg_attraction,
    'code_usage': total_code_usage,
    'reward': avg_reward
}]
```

## Result

- Memory usage stays constant (~23 GB) throughout full pretrain runs
- EWA metrics still available for online-iteration plots
- No history or trajectory data accumulation across batches

## Key lesson

ML training memory leaks often come from accumulating detailed per-step/per-batch data in loops, not from DataLoader configuration. Always audit what data structures grow with training progress.
