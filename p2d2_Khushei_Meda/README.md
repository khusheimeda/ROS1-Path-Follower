# Wall Follower RL — Deliverable 2

**Package:** `wall_follower_rl`  
Q-Learning and SARSA wall-following for the Triton robot in Gazebo (ROS1 Noetic).

---

# Requirements

- ROS Noetic on **Ubuntu 20.04**
- `stingray_sim` package (place parallel to this package in `catkin_ws/src/`)
- Python packages:
  - `numpy`

---

# How to Run

## Build

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

# Training

## Q-Learning training
roslaunch wall_follower_rl wall_follower.launch algorithm:=qlearning mode:=train

## SARSA training
roslaunch wall_follower_rl wall_follower.launch algorithm:=sarsa mode:=train

---

# Testing

## Q-Learning
roslaunch wall_follower_rl wall_follower.launch algorithm:=qlearning mode:=test

# SARSA
roslaunch wall_follower_rl wall_follower.launch algorithm:=sarsa mode:=test

---

# Override Hyperparameters from Command Line

```bash
roslaunch wall_follower_rl wall_follower.launch algorithm:=qlearning mode:=train \
  num_episodes:=500 epsilon_start:=0.90 alpha:=0.10
```

---

# Teleport the Robot During Testing

```bash
rosservice call /gazebo/set_model_state "model_state:
  model_name: 'triton'
  pose:
    position: {x: 0.4, y: -1.26, z: 0.00}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  reference_frame: 'world'"
```

---

# Design

## State Representation

The LiDAR scan is divided into **3 sectors**, each discretized into **3 distance bins**, yielding **3³ = 27 discrete states**.

### Sector Definitions

| Sector | Angle Range | Purpose |
|------|------|------|
| front | -20° to +20° | Obstacle detection ahead |
| front_right | -60° to -20° | Upcoming wall / corner detection |
| right | -110° to -60° | Primary wall-tracking sensor |

### Distance Bins

| Bin | Range | Meaning |
|----|----|----|
| close | < 0.40 m | Too near / danger |
| desired | 0.40 – 0.90 m | Sweet spot (d_w ≈ 0.75 m) |
| far | ≥ 0.90 m | Wall lost / open space |

---

# Action Space

| Index | Linear (m/s) | Angular (rad/s) | Description |
|----|----|----|----|
| 0 | 0.02 | +1.20 | Hard-left |
| 1 | 0.12 | +0.50 | Soft-left |
| 2 | 0.22 | 0.00 | Straight |
| 3 | 0.12 | -0.50 | Soft-right |
| 4 | 0.02 | -1.20 | Hard-right |

---

# Reward Function

| Condition | Reward |
|------|------|
| Collision (any sector < 0.18 m) | -100 (episode ends) |
| Right wall in desired zone | +15.0 × linear_vel |
| Right wall too close | -8.0 |
| Right wall too far | -10.0 |
| Front obstacle < 0.35 m | -20.0 |
| Front obstacle < 0.55 m | -8.0 |
| Forward progress bonus | +4.0 × linear_vel |
| Spinning penalty | -3.0 × |
| Living penalty | -0.5 |

---

# Q-Table Initialization

Q-values are initialized with **expert knowledge from Deliverable 1 (manual Q-table)** rather than zeros.

This encodes basic wall-following heuristics:

- Right wall in **desired range** favors **going straight**
- Right wall **too close** favors **turning left**
- Right wall **too far / open space** favors **turning right or going straight**

Q-learning and SARSA then update these values through experience.

---

# Hyperparameters

## Q-Learning

| Parameter | Episodes 1–300 | Episodes 301–400 |
|------|------|------|
| α (alpha) | 0.10 | 0.10 |
| γ (gamma) | 0.95 | 0.95 |
| ε start | 0.90 | 0.30 |
| ε min | 0.05 | 0.05 |
| ε decay | 0.997 | 0.99 |

At **episode 300**, ε was reduced to fine-tune the policy with less exploration.

---

## SARSA

| Parameter | Value |
|------|------|
| α (alpha) | 0.10 |
| γ (gamma) | 0.95 |
| ε start | 0.50 |
| ε min | 0.05 |
| ε decay | 0.99 |
| Episodes | 300 |

SARSA was trained with a **lower initial ε (0.50 vs 0.90)** because, as an **on-policy algorithm**, high exploration rates directly corrupt Q-value estimates.

Random exploratory actions feed into the TD target, unlike Q-learning which always updates toward the best possible action regardless of what was actually taken.

---

# Training Spawns

The robot is spawned at varied positions each episode to ensure coverage of all **five wall-following scenarios**.

| Category | Count | Purpose |
|------|------|------|
| Straight walls | 4 | Basic wall tracking |
| Corners | 4 | Inside and outside 90° turns |
| I-beam | 6×2 | Thin freestanding wall (double-weighted) |
| Open space | 4 | Learning to find a wall from origin |

The **I-beam spawns are double-weighted** to ensure the robot adequately learns to follow both faces and navigate around the tips of the thin freestanding wall.

---

# Algorithm Comparison

## Convergence

SARSA converged significantly faster than Q-learning.

- **SARSA:** near-zero accumulated reward by approximately **episode 50–75**
- **Q-learning:** required roughly **250–300 episodes**

Even after fine-tuning Q-learning with reduced exploration (ε lowered at episode 300), its accumulated rewards remained slightly below those of SARSA.

Part of this difference is attributable to SARSA being trained with a **lower initial ε (0.50 vs 0.90)**, meaning less time was spent on random exploration. However, this choice was itself a consequence of the algorithmic difference: SARSA requires lower exploration because it is **on-policy**. Exploratory (random) actions directly influence the TD update target, which can push Q-values in unproductive directions. Q-learning is **off-policy** and more robust to high exploration, but in this compact state space (27 states), that robustness was not advantageous enough to overcome the slower convergence caused by spending more episodes exploring randomly.

---

## Policy Quality

Both algorithms produce policies that successfully navigate all five scenarios:

- Straight walls  
- Inside corners  
- Outside corners  
- I-beam faces  
- Open space  

However, visual evaluation revealed a qualitative difference.

### Q-Learning Behavior

Q-Learning exhibits a recurring **Z-shaped oscillation** along every straight wall segment. The robot periodically drifts to the boundary of the desired distance bin, briefly enters the **far** bin, corrects with a right turn, then overshoots back into **desired**, producing a visible zigzag.

This occurs on every wall of the maze perimeter.

### SARSA Behavior

SARSA produces **smoother wall-following**, with the Z-oscillation appearing on only **one wall segment rather than all of them**. The robot maintains a more consistent distance from the wall with fewer unnecessary corrections.

---

# Conclusion

**SARSA produced the better policy for this task.**

It:

- Converged faster
- Achieved higher accumulated rewards during training
- Generated smoother wall-following behavior during testing

The **compact state space (27 states)** and the **continuous nature of the wall-following task** favor SARSA’s **conservative on-policy learning** over Q-learning’s **optimistic off-policy updates**.