# Cognitive Architecture for an Autonomous Robot in Robobo Sim

## Project Overview
This project is part of the *Intelligent Robotics II* course for the academic year 2023/2024. It focuses on designing and implementing a **Cognitive Architecture** for an autonomous robot in the **Robobo Simulator** using **Python** and relevant libraries.

## Authors
- **Ettore Caputo**
- **Stefano Iannicelli**


## Introduction
This project aims to develop a **cognitive architecture** that enables a robot to autonomously navigate and achieve its goals in a simulated environment. The architecture is based on a **deliberative decision system**, incorporating a **World Model** and a **Utility Model**.

## Problem Description
The challenge is to design a system that allows the robot to process sensor data, predict future states, and make decisions to reach its objectives autonomously.

## Action Execution
The robot's movements are controlled using its **gyroscope sensor**, with rotations calculated using specific mathematical formulas.

## Proposed Solution
The proposed solution consists of two main components:

### World Model
The **World Model** predicts the robot’s future states based on its current state and actions.

#### Mathematical Version
- Uses algebraic operations to estimate future positions.
- Accounts for sensor data to update the state.

#### Neural Network Version
- Implements two neural networks:
  - One for **position prediction**.
  - One for **sensor data prediction**.
- Trained to generalize physical behavior in the simulation.

### Utility Model
The **Utility Model** helps in decision-making using two components:

#### Intrinsic Module
- Implements a **novelty function** to encourage exploration.
- Computes novelty values based on past experiences.

#### Extrinsic Module
- Uses a **neural network** to evaluate predicted states.
- Utilizes a **Binary Cross-Entropy loss function** for training.
- Rewards states closer to the goal with higher values.

## Training Strategy
- Uses an **ϵ-greedy policy** for learning.
- Initially selects actions based on **novelty**.
- Over time, decision-making shifts towards the **neural network**.
- With the **mathematical World Model**, the robot learns a policy in approximately **10 minutes**.

## Challenges and Solutions
- The **neural World Model** struggled to predict the next states accurately.
- Issue likely due to incorrect **object size prediction**.
- Resulted in erratic behavior where the robot avoided obstacles but failed to move forward.
- Future improvements could focus on refining **object perception mechanisms**.

## Conclusions
- A **simple cognitive architecture** was successfully implemented.
- The **mathematical World Model** provided effective results, enabling the robot to learn goal-directed behavior after an initial exploration phase.
- The **neural World Model** requires further refinement for better performance.



## Future Work
- Improve the **neural World Model** for better prediction accuracy.
- Experiment with different **reward functions**.
- Test the system in more complex **simulated environments**.

---

For more details, refer to the **project report**.
