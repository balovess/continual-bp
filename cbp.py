import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

class ContinualBackpropagation:
    def __init__(self, model, optimizer=None, reset_rate=0.01, maturity_threshold=50, utility_decay=0.05,
                 reset_init='uniform', momentum=0.9, device=None):
        """
        Initializes the Continual Backpropagation (CBP) algorithm.

        Args:
            model: The PyTorch model.
            optimizer: The optimizer (optional).
            reset_rate: The proportion of weights to reset in each reset operation.
            maturity_threshold: The maturity threshold for weights.
            utility_decay: The decay factor for utility values.
            reset_init: The initialization method for reset weights ('uniform', 'kaiming', 'orthogonal', 'normal', 'constant').
            momentum: Momentum for smoothing utility value changes.
            device: The device to run the computations on (e.g., 'cuda', 'cpu').
        """
        self.model = model
        self.optimizer = optimizer
        self.reset_rate = reset_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.reset_init = reset_init
        self.momentum = momentum
        self.device = device
        # Initialize state
        self.utility = {}
        self.maturity = {}
        self.is_frozen = {}

        # Initialize per-layer parameter blocks
        for name, param in model.named_parameters():
            if 'weight' in name:  # Track weights only
                # Use float16 to reduce memory usage
                self.utility[name] = torch.zeros_like(param, dtype=torch.float16, device=param.device)
                self.maturity[name] = torch.zeros_like(param, dtype=torch.int16, device=param.device)
                self.is_frozen[name] = torch.zeros_like(param, dtype=torch.bool, device=param.device)

    def _reset_weights(self, param, indices):
        """Resets the selected weights using the specified initialization strategy."""
        with torch.no_grad():  # Disable gradient tracking
            flat_param = param.view(-1)
            if self.reset_init == 'uniform':
                new_values = (torch.rand(indices.size(), device=param.device) - 0.5) * 0.1
            elif self.reset_init == 'kaiming':
                if param.dim() < 2:
                    new_values = torch.randn(indices.size(), device=param.device) * 0.1
                else:
                    new_values = torch.empty(indices.size(), device=param.device)
                    init.kaiming_uniform_(new_values, nonlinearity='relu')
            elif self.reset_init == 'orthogonal':
                if param.dim() < 2:
                    raise ValueError("Orthogonal initialization requires at least 2D tensors.")
                else:
                    new_values = torch.empty(indices.size(), device=param.device)
                    init.orthogonal_(new_values.view(-1, 1))
            elif self.reset_init == 'normal':
                new_values = torch.randn(indices.size(), device=param.device) * 0.1
            elif self.reset_init == 'constant':
                new_values = torch.zeros(indices.size(), device=param.device)
            else:
                raise ValueError(f"Unsupported reset_init type: {self.reset_init}")

            flat_param[indices] = new_values

    def _update_utility(self):
        """Updates the utility values of weights based on gradient contribution."""
        for name, param in self.model.named_parameters():
            if param.grad is None or 'weight' not in name:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8).to(param.device)
            if self.optimizer:
                state = self.optimizer.state[param]
                # Ensure exp_avg_sq exists and is a tensor
                exp_avg_sq = state.get('exp_avg_sq', torch.ones_like(param, device=param.device))
                momentum_buffer = state.get('momentum_buffer', torch.zeros_like(param, device=param.device))
                grad_contribution *= (torch.abs(momentum_buffer) + 1e-8)
                grad_contribution /= (torch.sqrt(exp_avg_sq) + 1e-8)

            # Update utility values
            self.utility[name] = self.utility_decay * self.utility[name].to(self.device) + (1 - self.utility_decay) * grad_contribution.to(self.device)
            # Regularly clean up state information
            self.utility[name] = torch.clamp(self.utility[name], min=0, max=1e6)  # Limit utility value range

    def _selective_reset(self):
        """Selectively resets low-utility weights and freezes high-utility weights."""
        for name, param in self.model.named_parameters():
            if 'weight' not in name or name not in self.utility:  # Process weights only
                continue

            utility = self.utility[name].flatten().to(param.device)
            maturity = self.maturity[name].flatten().to(param.device)
            is_frozen = self.is_frozen[name].view(-1).to(param.device)

            # Freeze high-utility weights
            freeze_threshold = max(torch.quantile(utility, 0.9), 1e-6)  # Avoid freezing failure due to all zeros
            is_frozen[utility > freeze_threshold] = True
            # Remove state after freezing
            if torch.all(is_frozen):
                self.utility[name] = torch.zeros_like(param, dtype=torch.float16, device=param.device)
                self.maturity[name] = torch.zeros_like(param, dtype=torch.int16, device=param.device)
                self.is_frozen[name] = torch.zeros_like(param, dtype=torch.bool, device=param.device)
                continue
            # Filter resettable weights
            valid_mask = (~is_frozen) & (maturity >= self.maturity_threshold)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            # Dynamically determine the number of weights to reset
            num_to_reset = min(int(self.reset_rate * utility.numel()), valid_indices.numel())
            if num_to_reset > 0:
                reset_indices = valid_indices[torch.randperm(valid_indices.numel())[:num_to_reset]]
                self._reset_weights(param, reset_indices)
                utility[reset_indices] = 0
                maturity[reset_indices] = 0
            # Update and store back the states
            self.utility[name] = utility.view_as(param)
            self.maturity[name] = maturity.view_as(param)
            self.is_frozen[name] = is_frozen.view_as(param)
                def _increment_maturity(self):
        """Dynamically increments maturity based on gradient contribution."""
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.maturity:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8)

            # Use median instead of mean for robustness to outliers
            median_contribution = torch.median(grad_contribution)

            # Use interquartile range (IQR) instead of standard deviation for robustness
            q75, q25 = torch.quantile(grad_contribution, 0.75), torch.quantile(grad_contribution, 0.25)
            iqr_contribution = q75 - q25

            # Dynamic threshold: median + 1.5 * IQR. This makes maturity increase only for significantly large gradient contributions.
            dynamic_threshold = median_contribution + 1.5 * iqr_contribution

            # Increment maturity only where gradient contribution exceeds the dynamic threshold.
            self.maturity[name] += (grad_contribution > dynamic_threshold).long()

    def step(self, current_step, step_interval=1):
        """Executes a single update operation of CBP."""
        if current_step % step_interval == 0:
            self._update_utility()
            self._selective_reset()
            self._increment_maturity()
            # torch.cuda.empty_cache() #Consider using this if you encounter memory issues.
