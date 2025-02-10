# continual-bp

**PyTorch Implementation of Continual Backpropagation (CBP)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements the Continual Backpropagation (CBP) algorithm using PyTorch to address catastrophic forgetting in continual learning. CBP maintains network plasticity by selectively resetting units (neurons or parameters) with low contribution utility and high maturity, enabling the network to learn new tasks while retaining knowledge from previous ones.

## Core Idea

Traditional deep learning methods are prone to catastrophic forgetting in continual learning because they are typically trained on a fixed dataset and then deployed. When the model encounters new data from subsequent tasks, it tends to adjust its weights to fit the new data, potentially overwriting previously learned knowledge.

The core principles behind CBP are:

*   **Utility:** Measures the contribution of a unit (neuron or parameter) to the model's performance. Units with lower utility have a smaller impact and are therefore more likely to be reset. Utility is often calculated as the product of the absolute value of the gradient and the absolute value of the weight (`torch.abs(param.grad) * torch.abs(param)`).
*   **Maturity:** Prevents newly initialized units from being reset too quickly. Only units that have undergone a certain number of updates are considered for resetting. This is typically tracked using a counter for each unit.
*   **Selective Resetting:** Resets only a portion of the units with the lowest utility, rather than resetting all of them, to maintain model stability. The reset probability is usually a hyperparameter.

By continuously performing selective resetting, CBP enables the network to maintain plasticity, allowing it to better adapt to new tasks while retaining old knowledge.


## References
Dohare et al. (2024): Loss of Plasticity in Deep Continual Learning (https://arxiv.org/abs/2306.13812)

## Original Implementation
The original implementation of Continual Backpropagation can be found at:

https://github.com/shibhansh/loss-of-plasticity

## Installation

```bash
pip install torch
