# 🧠🤖 Cognitive Architecture for an Autonomous Robot in Robobo Sim

## 🌟 Project Overview
This project is part of the *Intelligent Robotics II* course for the academic year **2023/2024**. We focused on designing and implementing a **Cognitive Architecture** for an autonomous robot in the **Robobo Simulator**, using **Python** and some awesome libraries! 🐍

##  Authors
- **Ettore Caputo**   
- **Stefano Iannicelli** 

## 🚀 Introduction
Our goal was to build a smart robot that could **navigate autonomously** and make decisions on its own in a simulated world 🌍🛣️. We designed a **deliberative decision-making system** that includes a **World Model** and a **Utility Model**.

## 🤔 Problem Description
How can a robot understand the world around it, predict what will happen next, and choose the best thing to do — all on its own? That's the challenge we tackled! 🔍⚙️

## 🕹️ Action Execution
The robot moves using its **gyroscope sensor** 🔄. Rotations are calculated with specific math formulas to guide its motion precisely. 📐🤖

## 🧩 Proposed Solution

### 🌐 World Model
This part predicts the robot’s future state based on its current situation and actions. It comes in two flavors:

#### 🔢 Mathematical Version
- Uses good ol’ **algebra** to estimate future positions 
- Integrates **sensor data** to update its understanding of the world

#### 🧠 Neural Network Version
- Includes **two neural networks**:
  - One for **position prediction** 
  - One for **sensor data prediction**   
- Learns to generalize the robot’s behavior from experience 

### 🎯 Utility Model
Helps the robot decide what to do next by evaluating how "good" different future states are.

#### 🔍 Intrinsic Module
- Encourages the robot to explore new things using a **novelty function**   
- Tracks past experiences to measure how “new” something is

#### 🏆 Extrinsic Module
- Uses a **neural network** to evaluate predicted states   
- Trained with **Binary Cross-Entropy loss**  
- Rewards states that bring the robot **closer to its goal** 

## 🧪 Training Strategy
- Follows an **ϵ-greedy policy** for choosing actions   
- Starts off curious (focused on **novelty**)   
- Gradually learns from experience using the **neural network**   
- With the **mathematical model**, it can learn a good policy in just **~10 minutes**! ⏱

## 🛠️ Challenges and Solutions
- The **neural World Model** had trouble accurately predicting future states 
- One issue was likely **incorrect object size perception**   
- This led to strange behavior — the robot avoided obstacles but didn’t go anywhere   
- Future improvements should aim to **refine object perception** and improve model accuracy 

## ✅ Conclusions
- We built a **simple but effective cognitive architecture** 
- The **mathematical World Model** enabled the robot to learn purposeful behavior after a short exploration phase 
- The **neural version** shows promise but needs more training and tuning 

## 🔮 Future Work
- Improve the **neural World Model**   
- Try out different **reward functions** to fine-tune learning 
- Test in more **complex environments** to push the limits 

---

📄 For more details, check out the **project report**! Thanks for reading! 😊🚀

