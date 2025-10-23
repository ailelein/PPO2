# PPO2

A simplified and custom implementation of the Proximal Policy Optimization (PPO) algorithm in Python, intended for training UAV systems and other reinforcement learning environments.

---

## 📖 Table of Contents

- [About](#about)  
- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
  - [Running training](#running-training)  
  - [Inference / evaluation](#inference-evaluation)  
- [Project Structure](#project-structure)  
- [Parameters](#parameters)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## 💡 About

This repository implements a version of the Proximal Policy Optimization (PPO) algorithm designed to work with UAV (Unmanned Aerial Vehicle) control tasks and other custom reinforcement-learning environments. It provides modules for environment interaction, policy/value networks, training and evaluation.  

The goal is to offer a self-contained framework to experiment with PPO on aerial robotics or other state-space / action-space domains.

---

## ✨ Features

- PPO algorithm implementation in Python  
- Custom modules for UAV control and trajectory planning  
- Support for multiple environment types (see `UAVxyz.py`, etc.)  
- Logging and plotting utilities included  
- Parameter file for easy configuration (`parameters.py`)  
- Script variants for “with MUAVs” / “without MUAVs” scenarios  

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.x  
- Standard ML / RL libraries: e.g., `numpy`, `matplotlib`, `gym` (if using OpenAI Gym)  
- (Optionally) GPU support if using deep networks  

### Installation

```bash
# Clone the repository
git clone https://github.com/ailelein/PPO2.git
cd PPO2

# Install dependencies (example)
pip install -r requirements.txt

