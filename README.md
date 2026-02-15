# Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning

This repository contains the official implementation of the **ICLR 2026** paper:

**Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning**

📄 **Paper (OpenReview):**  
https://openreview.net/forum?id=7N0ugLE17t

---

## Overview

This codebase implements a **Continuous-Time Value Iteration** framework for **Multi-Agent Reinforcement Learning (MARL)**.

The key idea is to perform value learning directly in continuous time under an HJB-style formulation, and to enable stable learning through **Value Gradient Iteration (VGI)** and continuous-time-consistent objectives.

The implementation focuses on:
- continuous-time multi-agent reinforcement learning,
- continuous-time value learning under HJB consistency,
- value gradient iteration (VGI),
- stable policy optimization in continuous-time settings.

---

## Repository Structure

The repository is organized as follows:

```
VIP_CTMARL/
├── algo/
│   ├── vip/
│   ├── ma_main.py              # off-the-shell code
│   │   ├── agent.py            # Core CT-VI/VIP agent implementation
│   │   ├── network.py          # Neural network modules (policy, value, dynamics, reward)
├── continuous_env/              # Continuous-time multi-agent environments
├── requirements.txt
└── README.md
```

---

## Code Description

### `ma_main.py`

`ma_main.py` serves as an **off-the-shelf training script** for running all experiments.

It is responsible for:
- argument parsing,
- environment initialization,
- training loop execution,
- logging and model checkpointing.

This file is the **main entry point** for reproducing experimental results.

---

### `algo/vip/agent.py`

This file contains the **core implementation of the proposed method**, including:
- continuous-time value learning (CT-VI) under HJB-style objectives,
- value gradient iteration (VGI),
- learning-based dynamics and reward (cost) models (when enabled),
- policy optimization via Hamiltonian-based objectives.


---

## Installation

We recommend using a virtual environment.

Install all required dependencies via:

```bash
pip install -r requirements.txt
```

---

## Running an Example

All experiments are launched through `ma_main.py`.

### Example Command

```bash
python main.py --algo vip --scenario target --seed 113
```


## Reference

If you find this repository useful, please consider citing our paper:

**Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning**  
ICLR 2026  

OpenReview:  
https://openreview.net/forum?id=7N0ugLE17t

---
