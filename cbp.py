import torch
import torch.nn.init as init

class MemoryPool:
    def __init__(self):
        """
        Memory pool to avoid repeated memory allocation.
        """
        self.pool = {}

    def allocate(self, shape, dtype=torch.float16, device=None):
        """
        Allocate memory from the pool or create new memory if not available.

        Args:
            shape (torch.Size): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device of the tensor.

        Returns:
            torch.Tensor: Allocated tensor.
        """
        key = (shape, dtype, device)
        if key not in self.pool:
            self.pool[key] = torch.zeros(shape, dtype=dtype, device=device)
        return self.pool[key]

class ContinualBackpropagation:
    def __init__(self, model, optimizer=None, reset_rate=0.01, maturity_threshold=50, utility_decay=0.05,
                 reset_init='uniform', momentum=0.9, device=None):
        """
        Initialize the Continual Backpropagation (CBP) algorithm.

        Args:
            model (torch.nn.Module): PyTorch model.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to None.
            reset_rate (float, optional): Proportion of weights to reset per step. Defaults to 0.01.
            maturity_threshold (int, optional): Maturity threshold for resetting weights. Defaults to 50.
            utility_decay (float, optional): Decay factor for utility values. Defaults to 0.05.
            reset_init (str, optional): Initialization method for reset weights ('uniform', 'orthogonal', 'normal', 'constant'). Defaults to 'uniform'.
            momentum: Momentum for smoothing utility value changes. Defaults to 0.9.
            device (torch.device, optional): Device to run on. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.reset_rate = reset_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.reset_init = reset_init
        self.momentum = momentum

        # Create memory pool
        self.memory_pool = MemoryPool()

        # Allocate state tensors using the memory pool
        self.state = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.state[name] = {
                    'utility': self.memory_pool.allocate(param.shape, dtype=torch.float16, device=param.device),
                    'maturity': self.memory_pool.allocate(param.shape, dtype=torch.int16, device=param.device),
                    'is_frozen': self.memory_pool.allocate(param.shape, dtype=torch.bool, device=param.device)
                }

    def _reset_weights(self, param, indices):
        """Reset selected weights using the specified initialization strategy."""
        with torch.no_grad():  # Disable gradient tracking
            flat_param = param.view(-1)
            if self.reset_init == 'uniform':
                new_values = (torch.rand(indices.size(), device=param.device) - 0.5) * 0.1
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
        """Update the utility values of weights based on gradient contribution."""
        for name, param in self.model.named_parameters():
            if param.grad is None or 'weight' not in name:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8).to(param.device)

            # Smooth utility with momentum if optimizer is available and has state
            if self.optimizer and param in self.optimizer.state: # Check if param has state
                state = self.optimizer.state[param]
                exp_avg_sq = state.get('exp_avg_sq', torch.ones_like(param, device=param.device))
                momentum_buffer = state.get('momentum_buffer', torch.zeros_like(param, device=param.device))
                grad_contribution *= (torch.abs(momentum_buffer) + 1e-8)
                grad_contribution /= (torch.sqrt(exp_avg_sq) + 1e-8)

            utility = self.state[name]['utility']
            utility = self.utility_decay * utility + (1 - self.utility_decay) * grad_contribution
            self.state[name]['utility'] = torch.clamp(utility, min=0, max=1e6)

    def _selective_reset(self):
        """Selectively reset low-utility weights and freeze high-utility weights."""
        for name, param in self.model.named_parameters():
            if 'weight' not in name or name not in self.state:
                continue

            utility = self.state[name]['utility'].flatten().to(param.device)
            maturity = self.state[name]['maturity'].flatten().to(param.device)
            is_frozen = self.state[name]['is_frozen'].view(-1).to(param.device)

            # Freeze high-utility weights
            freeze_threshold = max(torch.quantile(utility, 0.9), 1e-6)
            is_frozen[utility > freeze_threshold] = True

            # Select resettable weights
            valid_mask = (~is_frozen) & (maturity >= self.maturity_threshold)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            # Dynamically determine the number of weights to reset
            num_to_reset = min(int(self.reset_rate * utility.numel()), valid_indices.numel())
            if num_to_reset > 0:
                reset_indices = valid_indices[torch.randperm(valid_indices.numel())[:num_to_reset]]
                self._reset_weights(param, reset_indices)
                utility[reset_indices] = 0
                maturity[reset_indices] = 0

            # Update state
            self.state[name]['utility'] = utility.view_as(param)
            self.state[name]['maturity'] = maturity.view_as(param)
            self.state[name]['is_frozen'] = is_frozen.view_as(param)

    def _increment_maturity(self):
        """Dynamically increment maturity based on gradient contribution."""
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.state:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8)

            # Use median and IQR for dynamic threshold
            median_contribution = torch.median(grad_contribution)
            q75, q25 = torch.quantile(grad_contribution, 0.75), torch.quantile(grad_contribution, 0.25)
            iqr_contribution = q75 - q25
            dynamic_threshold = median_contribution + 1.5 * iqr_contribution

            # Increment maturity
            self.state[name]['maturity'] += (grad_contribution > dynamic_threshold).long()

    def step(self, current_step, step_interval=1):
        """Perform a single CBP update operation."""
        if current_step % step_interval == 0:
            self._update_utility()
            self._selective_reset()
            self._increment_maturity()
