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

## 🚀 Code

### 1. Modify `train.py`

Replace the real dataloader with synthetic data generation by editing the main training loop:

<pre><code class="language-python"># Replace this line inside TrainAgent.run()
for batch in self.train_dataloader:

# With:
batch = generate_fake_batch()
</code></pre>

### 2. Freeze VLM and Train Only the Action Expert

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

## 🧪 Finetune with SLURM Script

To execute the training locally using RTX 3090 GPU:

```bash
bash slurm/test_training_single_gpu_no_slurm.sh
```


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

### Training Visualization

This figure shows the training result using fake data and VLM frozen.

![Training visualization](image_0.png)