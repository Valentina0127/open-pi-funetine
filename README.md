# open-pi-zero

## 🎯 Fake Data Fine-Tuning for PiZero

This guide shows how to run the PiZero training script using randomly generated fake data to validate the training pipeline.

---

## 🧪 Experimental Setup

- **GPU**: NVIDIA RTX 3090  
- **PyTorch Version**: 2.1.0+cu118  
- **CUDA Version**: 11.8  
- **Python Version**: 3.10  
- **OS**: Ubuntu 20.04  

---

### 🛠️ Enable Fake Data Generation

To use synthetic data instead of real datasets during training, set the following in your config YAML file (e.g., `funetine.yaml`):

```yaml
data:
  train:
    dataset_mix: random
```
## 🧪 Finetune with SLURM Script

To execute the training locally using RTX 3090 GPU:

```bash
bash slurm/test_training_single_gpu_no_slurm.sh
```



## 🚀 Code

### 1. Modify `train.py`

Replace the real dataloader with synthetic data generation by editing the main training loop:

<pre><code class="language-python"># Replace this line inside TrainAgent.run()
for batch in self.train_dataloader:

# With:
batch = generate_fake_batch()
</code></pre>

### 1. Freeze VLM and Train Only the Action Expert

To only fine-tune the **Action Expert**, and freeze the VLM, update your config (e.g., `fractal.yaml`) with:

<pre><code class="language-yaml">train_vlm: False
</code></pre>

This ensures VLM parameters will not be updated during training.

---

## ✅ Log Verification

After running the training script, confirm in the logs that **only the action branch is being trained**:

<pre><code>[2025-05-08 12:43:24,078][src.agent.train][INFO] - Number of trained parameters (Action): 0.315B
</code></pre>

You will not see

<pre><code>[2025-05-08 12:43:24,081][src.agent.train][INFO] - Number of trained parameters (VLM): 2.291B
</code></pre>


---



## 🧩 Synthetic Data Format

The data format is reverse-engineered from the `TrainAgent.run()` function in `src/agent/train.py`, specifically from the inline comment:

```python
"""
batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 
             'image_wrist', 
             'timestep' (torch.Size([bsz, 1])), 
             'pad_mask_dict', 
             'timestep_pad_mask', 
             'task_completed' (torch.Size([bsz, window, 4]), 
             'proprio' (torch.Size([bsz, window, proprio_dim])
task: 'language_instruction', 
      'pad_mask_dict', 
      'image_primary', 
      'image_wrist', 
      'timestep' (torch.Size([bsz]))
action (torch.Size([bsz, window, horizon, action_dim], float32)
action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
"""
```
The following function is used to generate such fake data:
```python
def generate_fake_batch(batch_size=8, window=1, horizon=4, action_dim=7, proprio_dim=8, height=224, width=224):
    return {
        "observation": {
            "image_primary": torch.randint(0, 256, (batch_size, window, height, width, 3), dtype=torch.uint8),
            "proprio": torch.randn(batch_size, window, proprio_dim),
            "timestep": torch.randint(0, 100, (batch_size, window), dtype=torch.int32),
            "pad_mask_dict": {
                "image_primary": torch.ones(batch_size, window, dtype=torch.bool),
                "proprio": torch.ones(batch_size, window, dtype=torch.bool),
                "timestep": torch.ones(batch_size, window, dtype=torch.bool),
            },
            "timestep_pad_mask": torch.ones(batch_size, window, dtype=torch.bool),
            "task_completed": torch.randint(0, 2, (batch_size, window, horizon), dtype=torch.bool),
        },
        "task": {
            "language_instruction": [
                random.choice([
                    b"pick banana from white bowl",
                    b"place green can",
                    b"close middle drawer",
                    b"move redbull near chip bag",
                ]) for _ in range(batch_size)
            ]
        },
        "action": torch.randn(batch_size, window, horizon, action_dim),
        "action_pad_mask": torch.ones(batch_size, window, horizon, action_dim, dtype=torch.bool),
        "dataset_name": ["dummy_dataset"] * batch_size,
    }
```

### Training Visualization

This figure shows the training result using fake data and VLM frozen.

![Training visualization](image_0.png)