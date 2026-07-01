<div align="center">
    
# 🤖 REAL: Robust Extreme Agility Learning

### Robust Extreme Agility via Spatio-Temporal Policy Learning and Physics-Guided Filtering

**🏆 Accepted at IROS 2026**

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv%3A2603.17653-b31b1b.svg)](https://arxiv.org/abs/2603.17653)
[![Website](https://img.shields.io/badge/🌐_Website-REAL-blue.svg)](https://jialonglong.github.io/REAL_wb/)
[![License](https://img.shields.io/badge/📝_License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/🐕_Platform-Unitree%20Go2-blue.svg)](https://www.unitree.com/go2)

<br/>

<br/>

<img src="images/fig1_teaser.png" width="90%"/>

<p><i>
REAL enables a quadrupedal robot to chain highly dynamic parkour maneuvers across complex terrains<br/>
with nominal vision <b>(green box)</b>, and maintain stable locomotion even under severe visual degradation <b>(red box)</b>.
</i></p>

</div>

## 📋 Table of Contents

- [✨ Highlights](#-highlights)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Installation](#️-installation)
- [🚀 Training Pipeline](#-training-pipeline)
- [📊 Evaluation](#-evaluation)
- [📦 Deployment](#-deployment)
- [🧪 Testing](#-testing)
- [📈 Results](#-results)
- [🔖 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Highlights

| | |
|---|---|
| 🧠 **Spatio-Temporal Policy Learning** — A privileged teacher learns structured proprioception–terrain associations via cross-modal attention. The distilled student uses a FiLM-modulated Mamba backbone to suppress visual noise and build short-term terrain memory. | ⚛️ **Physics-Guided Filtering** — An uncertainty-aware neural velocity estimator is fused with rigid-body dynamics through an Extended Kalman Filter (EKF), ensuring physically consistent state estimation during impacts and slippage. |
| 🎯 **Consistency-Aware Loss Gating** — Adaptive gating between behavioral cloning and RL stabilizes policy distillation and improves sim-to-real transfer, preventing policy collapse under aggressive domain randomization. | ⚡ **Real-Time Onboard Deployment** — Bounded O(1) inference at ~13.1 ms/step on a Unitree Go2 with zero-shot sim-to-real transfer — no fine-tuning required on the real robot. |

---

## 🏗️ Architecture

<div align="center">
<img src="images/fig2_architecture.png" width="95%"/>
</div>

> **Stage 1 — Privileged Teacher Policy Learning:**
> The teacher policy learns precise proprioception–terrain associations through cross-modal attention. Proprioceptive states serve as Queries to selectively retrieve relevant terrain features encoded as Keys and Values from terrain scan dots.

> **Stage 2 — Distilling Student Policy with Spatio-Temporal Reasoning:**
> The deployable student integrates FiLM-based visual–proprioceptive fusion with a Mamba temporal backbone. A physics-guided Bayesian estimator and consistency-aware loss gating further stabilize training and deployment.

---

## ⚙️ Installation

### 📋 Prerequisites

- Ubuntu 18.04+ with NVIDIA GPU
- CUDA 11.3+
- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) (download separately)

### 🔧 Setup

```bash
# Create conda environment
conda create -n real python=3.8
conda activate real

# Install PyTorch (adjust CUDA version as needed)
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Isaac Gym (download from NVIDIA, then):
cd <isaacgym_dir>/python && pip install -e .

# Install REAL packages
cd REAL/rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .

# Additional dependencies
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python flask pymeshlab
```

---

## 🚀 Training Pipeline

REAL training follows a two-stage pipeline: first training a privileged teacher, then distilling into a deployable student with depth vision.

### 📂 Project Structure

```
REAL/
├── legged_gym/                  # Environment & scripts
│   └── legged_gym/
│       ├── envs/
│       │   ├── base/
│       │   │   ├── legged_robot.py          # Core simulation environment
│       │   │   └── legged_robot_config.py   # Default config (LeggedRobotCfg / LeggedRobotCfgPPO)
│       │   └── go2/
│       │       └── go2_config.py            # Go2-specific config overrides
│       ├── scripts/
│       │   ├── train.py                     # Main training entry point
│       │   ├── play.py                      # Visualize trained policy
│       │   ├── evaluate.py                  # Single-terrain evaluation
│       │   ├── evaluate_metrics.py          # Batch evaluation with metrics report
│       │   └── save_jit.py                  # Export to TorchScript for deployment
│       └── utils/
│           ├── helpers.py                   # CLI argument definitions
│           ├── terrain.py                   # Procedural terrain generation
│           └── task_registry.py             # Task registration & runner factory
├── rsl_rl/                      # RL algorithm & neural network modules
│   └── rsl_rl/
│       ├── algorithms/
│       │   └── ppo.py                       # PPO + Huber-Gaussian loss + KF fusion
│       ├── modules/
│       │   ├── actor_critic.py              # Actor, ActorCriticRMA (teacher)
│       │   ├── proprio_cross_attention_encoder.py  # Cross-modal attention (Eq. 1)
│       │   ├── depth_backbone.py            # FiLM-CNN + Mamba + GRU (student)
│       │   ├── mamba_block.py               # Selective SSM block (Eq. 3)
│       │   ├── estimator.py                 # MLP / TCN estimator
│       │   └── estimator_resnet1d.py        # 1D ResNet velocity estimator
│       ├── runners/
│       │   └── on_policy_runner.py          # Training loop orchestrator
│       └── storage/
│           └── rollout_storage.py           # PPO rollout buffer
└── tests/
    └── test_modules.py                      # CPU smoke tests (52 tests)
```

### 🎓 Stage 1: Privileged Teacher Training

The teacher has access to ground-truth terrain scan dots and privileged state information (velocity, friction, motor strength). It uses a **cross-modal attention encoder** (Eq. 1) where proprioceptive states serve as Queries and terrain features as Keys/Values.

```bash
python legged_gym/legged_gym/scripts/train.py \
    --task go2 \
    --exptid teacher-v1 \
    --max_iterations 20000
```

Key arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Registered task name | `go2` |
| `--exptid` | Experiment ID (auto-generated if omitted) | auto |
| `--max_iterations` | Total PPO iterations | `40000` |
| `--num_envs` | Number of parallel environments | `2048` |
| `--seed` | Random seed | `1` |
| `--resume` | Resume from latest checkpoint | `False` |
| `--resumeid` | Resume from specific experiment | — |
| `--debug` | Debug mode (64 envs, wandb disabled) | `False` |
| `--no_wandb` | Disable W&B logging | `False` |
| `--delay` | Enable action delay domain randomization | `False` |
| `--use_camera` | Enable depth camera (for Stage 2) | `False` |

Logs are saved to `/data/parkour_logs/<proj_name>/<exptid>/`.

### 👁️ Stage 2: Student Distillation with Depth Vision

The student policy replaces terrain scan dots with depth camera observations. It uses:
- **FiLM-modulated CNN** (Eq. 2) for spatial depth–proprioception fusion
- **Mamba temporal backbone** (Eq. 3) for short-term terrain memory
- **Consistency-aware loss gating** (Eq. 11-12) to balance BC and RL

```bash
python legged_gym/legged_gym/scripts/train.py \
    --task go2 \
    --exptid student-v1 \
    --use_camera \
    --resume \
    --resumeid teacher-v1 \
    --max_iterations 30000
```

The `--resume --resumeid teacher-v1` loads the teacher checkpoint. When `--use_camera` is set, the runner automatically enters the vision distillation loop:
1. Teacher generates target actions from privileged observations
2. Student observes depth images and proprioception
3. Loss = λ · L_RL + (1-λ) · L_BC, where λ adapts via consistency gating

### ⚙️ Configuration

All default hyperparameters are in `legged_robot_config.py`:

**Environment** (`LeggedRobotCfg`):
- `env.n_proprio = 53` — proprioceptive observation dimension
- `env.n_scan = 132` — terrain scan dots dimension
- `env.history_len = 10` — proprioceptive history frames
- `env.depth_history_len = 2` — depth image history frames

**Policy** (`LeggedRobotCfgPPO.policy`):
- `scan_encoder_type = "proprio_cross_attention"` — teacher's cross-modal attention
- `scan_attn_d_model = 128, heads = 4, layers = 2`

**Depth Encoder** (`LeggedRobotCfgPPO.depth_encoder`):
- `depth_encoder_type = "mamba"` — Mamba temporal backbone
- `depth_mamba_d_state = 128, layers = 2`
- `film_spatial_enabled = True` — FiLM modulation on conv1 and conv2
- `consistency_gating_k = 2.0, tau = 0.5` — loss gating parameters

**Estimator** (`LeggedRobotCfgPPO.estimator`):
- `model_type = "resnet1d"` — 1D ResNet velocity estimator
- `history_len = 10` — 10-frame proprioceptive sequence
- `uncertainty_enabled = True` — Huber-Gaussian loss (Eq. 4-5)
- `fusion_enabled = True` — EKF fusion (Eq. 6-10)

**PPO** (`LeggedRobotCfgPPO.algorithm`):
- `learning_rate = 1.5e-4`, `gamma = 0.99`, `lam = 0.95`
- `num_learning_epochs = 3`, `num_mini_batches = 4`

Task-specific overrides live in `go2/go2_config.py`.

### 💡 Training Tips

1. **Quick smoke test** — verify everything works before a full run:
   ```bash
   python legged_gym/legged_gym/scripts/train.py --task go2 --debug --max_iterations 3
   ```
   This uses 64 envs on flat terrain with wandb disabled.

2. **Action delay** — for sim-to-real robustness, enable after initial convergence:
   ```bash
   python legged_gym/legged_gym/scripts/train.py --task go2 --delay --resume --resumeid <exptid>
   ```

3. **Warm-start estimator** — pre-train the velocity estimator separately:
   ```bash
   python legged_gym/legged_gym/scripts/train.py --task go2 \
       --load_estimator_checkpoint /path/to/estimator.pt
   ```

4. **Monitor training** — use W&B (enabled by default in non-debug mode):
   ```bash
   wandb login
   python legged_gym/legged_gym/scripts/train.py --task go2 --exptid my-run
   ```

---

## 📊 Evaluation

### 🎮 Interactive Visualization

```bash
python legged_gym/legged_gym/scripts/play.py \
    --task go2 \
    --exptid <exptid>
```

### 📊 Batch Metrics Evaluation

```bash
python legged_gym/legged_gym/scripts/evaluate_metrics.py \
    --task go2 \
    --resumeid <exptid> \
    --num_trials 50 \
    --num_robots 20
```

This outputs a markdown report with per-terrain metrics: MXD (forward progress), MEV (edge violations), success rate, collision rate.

### 🌧️ Sensor Degradation Testing

Test policy robustness under perceptual corruption:

```bash
python legged_gym/legged_gym/scripts/evaluate_metrics.py \
    --task go2 \
    --resumeid <exptid> \
    --deg_enable \
    --deg_drop_prob 0.3 \
    --deg_noise_std 0.05
```

---

## 📦 Deployment

### 📤 Export to TorchScript

```bash
python legged_gym/legged_gym/scripts/save_jit.py \
    --exptid <exptid> \
    --run_dir /data/parkour_logs/parkour_new/<exptid>
```

This produces:
- `*-base_jit.pt` — policy network: `(obs, depth_latent) -> actions`
- `*-vision_jit.pt` — depth encoder (stateless)
- `*-vision_stateful_jit.pt` — depth encoder with explicit GRU hidden state
- `*-onboard_jit.pt` — combined estimator + history encoder + actor backbone
- `*-vision_weight.pt` — depth encoder state dict

### 🤖 Onboard Inference

On the robot (C++ with LibTorch):

```cpp
auto policy = torch::jit::load("base_jit.pt");
auto vision = torch::jit::load("vision_stateful_jit.pt");
torch::Tensor gru_hidden = torch::zeros({1, 1, 512});

// Control loop at 50 Hz
auto [depth_latent, new_hidden] = vision.forward({depth, proprio, gru_hidden});
gru_hidden = new_hidden;
auto actions = policy.forward({obs, depth_latent});
```

---

## 🧪 Testing

Run the CPU smoke test suite (52 tests, no GPU required):

```bash
python tests/test_modules.py -v
```

Test coverage:
- **Level 0**: Syntax check on all 43 .py files, config validation, module imports
- **Level 1**: Forward pass for every neural network component (Mamba, FiLM, CNN, cross-attention, estimators, ActorCritic)
- **Level 2**: PPO initialization, action sampling, single gradient update, Huber-Gaussian loss, KF fusion, consistency gating, JIT export wrappers

Run specific tests:

```bash
python tests/test_modules.py -k mamba     # Mamba-related only
python tests/test_modules.py -k ppo       # PPO-related only
python tests/test_modules.py -k jit       # JIT export only
```

---

## 📈 Results

### 🏔️ Extreme Terrain Traversability

| Method | Hurdles SR | Steps SR | Gaps SR | Overall SR | Overall MXD | MEV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Extreme Parkour | 0.18 | 0.14 | 0.10 | 0.16 | 0.21 | 34.24 |
| RPL | 0.05 | 0.04 | 0.03 | 0.04 | 0.10 | 1.56 |
| SoloParkour | 0.42 | 0.49 | 0.36 | 0.39 | 0.34 | 96.93 |
| **REAL (Ours)** | **0.82** | **0.94** | **0.28** | **0.78** | **0.45** | **18.41** |

### 🛡️ Robustness Under Perceptual Degradation

| Method | Nominal SR | Frame Drop SR | Gaussian Noise SR | FoV Occlusion SR |
|:---|:---:|:---:|:---:|:---:|
| Extreme Parkour | 0.16 | 0.16 (0.00) | 0.11 (-0.05) | 0.13 (-0.03) |
| SoloParkour | 0.39 | 0.20 (-0.19) | 0.37 (-0.03) | 0.41 (+0.02) |
| **REAL (Ours)** | **0.78** | **0.61** (-0.17) | **0.51** (-0.27) | **0.72** (-0.06) |

### 🔬 Component Ablation

| Variant | SR | MXD | MEV |
|:---|:---:|:---:|:---:|
| **REAL (Full)** | **0.78** | **0.45** | **18.41** |
| w/ MLP Estimator | 0.73 | 0.43 | 19.34 |
| w/o FiLM | 0.44 | 0.51 | 93.43 |
| w/o Mamba | 0.51 | 0.47 | 89.96 |

### 📐 Velocity Estimation

| Estimator | RMSE |
|:---|:---:|
| MLP (Baseline) | 0.52 |
| MLP + EKF | 0.40 |
| 1D ResNet (10 frames) | 0.28 |
| **1D ResNet + EKF (Ours)** | **0.23** |

---

## 🔖 Citation

```bibtex
@inproceedings{real2026,
  title     = {REAL: Robust Extreme Agility via Spatio-Temporal Policy Learning
               and Physics-Guided Filtering},
  author    = {Jialong Liu and Dehan Shen and Yanbo Wen
               and Zeyu Jiang and Changhao Chen},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.17653}
}
```

## 🙏 Acknowledgements

This work builds upon the simulation infrastructure of [Isaac Gym](https://developer.nvidia.com/isaac-gym) and the terrain setup from [Extreme Parkour](https://extreme-parkour.github.io/). We thank the authors for their open-source contributions.

## 📄 License

This project is released under the [MIT License](LICENSE).
