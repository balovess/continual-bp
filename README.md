# continual-bp

**PyTorch Implementation of Continual Backpropagation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements the Continual Backpropagation (CBP) algorithm using PyTorch, aiming to address the catastrophic forgetting problem encountered by neural networks in continual learning scenarios. CBP maintains network plasticity by selectively resetting neurons with lower contribution utility, enabling the network to learn new tasks without forgetting previously learned ones.

## Core Idea

Traditional deep learning methods are prone to catastrophic forgetting in continual learning because they are typically trained on a fixed dataset and then deployed. When the model is exposed to new data, it tends to adjust its weights to fit the new data, thereby overwriting previously learned knowledge.

The core ideas behind CBP are:

*   **Utility:** Measures the contribution of a neuron to the model's performance. Neurons with lower utility have a smaller impact on the model and are therefore more likely to be reset.
*   **Maturity:** Prevents newly initialized neurons from being reset too quickly. Only neurons that have undergone a certain number of updates are considered for resetting.
*   **Selective Resetting:** Resets only a portion of the neurons with the lowest utility, rather than resetting all of them, to maintain model stability.

By continuously performing selective resetting, CBP enables the network to maintain a degree of plasticity, allowing it to better adapt to new tasks while retaining old knowledge.

## Installation
