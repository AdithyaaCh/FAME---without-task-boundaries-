# Boundary-Free Continual Reinforcement Learning via Statistical and Implicit Task-Shift Detection in FAME

This repository contains the implementation of a **boundary-free extension of FAME (Fast and Meta Knowledge Learners for Continual Reinforcement Learning)** by replacing the assumption of known task boundaries with an online **Task-Shift Detection Module (TSDM)**.

In realistic environments, task switches are rarely announced. This project studies how an RL agent can autonomously detect environment changes and trigger knowledge transfer without access to an oracle boundary signal.

## Overview

Continual Reinforcement Learning (CRL) faces the stability–plasticity trade-off:

- **Plasticity:** adapt quickly to new tasks
- **Stability:** retain previously learned knowledge

FAME addresses this using a dual-learner architecture:

- **Fast Learner** → rapid adaptation
- **Meta Learner** → long-term knowledge consolidation

However, FAME assumes that task boundaries are known beforehand.

We remove this assumption by introducing a **Task-Shift Detection Module (TSDM)** that provides a binary trigger signal:



Environment → Detector → Task Shift → FAME Adaptation



We implement and compare three task-shift detectors:

1. Statistical detector (SWOKS-style)
2. Implicit detector (Task-Signature Network)
3. Hybrid detector (FSM + evidence fusion)

---

## Key Contributions

### 1. Statistical Task-Shift Detection (SWOKS)

A distribution-based detector using:

- Sliced Wasserstein Distance (SWD)
- Kolmogorov–Smirnov test
- Adaptive thresholding

The detector operates on latent action–reward features:

$$
d_t = [\phi_t, a_t, r_t/\sqrt{|\phi|}]
$$

where:

- \( \phi_t \) = Fast Learner latent representation
- \( a_t \) = action
- \( r_t \) = reward


### 2. Implicit Task-Signature Network (TSN)

A learned detector that predicts:

- future reward
- future latent dynamics

Architecture:


State Features + Action
|
MLP
/   
Reward    Dynamics
Head       Head


Detection is based on increases in prediction error using Welch's t-test.


### 3. Hybrid Detector

Combines both signals using a finite-state machine:

States:

```

Neutral → Suspect → Fire

```

Evidence fusion uses:

$$
-\log(p_{imp})-\log(p_{stat})
$$

to combine independent statistical evidence.

---

# Experimental Setup

## Environment

Benchmark:

**MinAtar Continual Reinforcement Learning**

Games:

- Breakout
- SpaceInvaders
- Freeway

Training:

- 7-task sequence
- 3.5M environment steps
- DQN backbone
- Kaggle T4 GPU


Task sequence:


Breakout
→ SpaceInvaders
→ Breakout
→ SpaceInvaders
→ SpaceInvaders
→ Freeway
→ Breakout


---

# Results

## Detection Performance

| Method | F1 Score | False Positives | Delay |
|--------|----------|-----------------|-------|
| Oracle | - | 0 | - |
| SWOKS | **0.91** | **0** | ~5k steps |
| Implicit | 0.12 | 56 | ~21k steps |
| Hybrid | 0.22 | 26 | ~13k steps |


SWOKS closely matches the oracle boundary detection while avoiding unnecessary resets.

---

## Game-Fair Performance

Average performance across unique games:

| Method | Avg Performance |
|--------|----------------|
| Oracle | 18.32 |
| SWOKS | **23.23** |
| Implicit | 15.41 |
| Hybrid | 16.33 |

SWOKS exceeds the oracle because it correctly avoids firing on same-task continuations.

Example:

The oracle resets at:


SpaceInvaders → SpaceInvaders


while SWOKS correctly continues training and accumulates additional experience.

---

## Retention

| Method | Retention |
|--------|-----------|
| Oracle | 1.000 |
| SWOKS | 1.000 |
| Implicit | 0.742 |
| Hybrid | 0.819 |

False-positive detections in implicit/hybrid methods caused unnecessary FAME resets and increased forgetting.

---


# Installation

Clone repository:

```bash
git clone <repo-url>
cd boundary-free-crl
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Experiments

Train with SWOKS detector:

```bash
python train.py --detector swoks
```

Train with implicit detector:

```bash
python train.py --detector implicit
```

Train with hybrid detector:

```bash
python train.py --detector hybrid
```

Oracle baseline:

```bash
python train.py --detector oracle
```

---

# Detector Interface

All detectors follow the same interface:

```python
detector.update(feature, action, reward)

if detector.shift_detected():
    trigger_FAME_update()
```

This allows plug-and-play replacement of the oracle boundary signal.

---

# Limitations

* Evaluated on a single MinAtar sequence
* Single random seed
* Limited benchmark diversity
* Implicit detector suffers from within-task representation drift
* Hyperparameters were not extensively tuned

Future work:

* Larger Atari benchmarks
* Meta-World environments
* Better drift modelling
* Learned snapshot selection

---

# Conclusion

This work demonstrates that continual RL can operate without explicit task boundaries by combining task-shift detection with FAME's knowledge transfer mechanism.

Among evaluated approaches:

* Statistical SWOKS detection provides the most reliable oracle replacement
* Learned implicit detection suffers from false positives
* Hybrid fusion improves over implicit detection but does not match statistical reliability

A carefully calibrated statistical detector is therefore a practical path toward boundary-free continual reinforcement learning.

---

## Authors

**Adithya Cheruvu**
**Diya Bhatnagar**
**Ekam Singh Sethi**

Indian Institute of Technology, Ropar

