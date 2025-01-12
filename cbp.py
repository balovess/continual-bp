import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class ContinualBackpropagation:
    def __init__(self, model, optimizer=None, reset_rate=0.01, maturity_threshold=50, utility_decay=0.05,
                 reset_init='uniform', enable_weighted_utility=True, reset_rates=None, momentum=0.9):
        self.model = model
        self.optimizer = optimizer
        self.reset_rate = reset_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.reset_init = reset_init
        self.enable_weighted_utility = enable_weighted_utility
        self.reset_rates = reset_rates if reset_rates is not None else {name: self.reset_rate for name, _ in model.named_parameters()}
        self.momentum = momentum

        # Initialize utility, momentum, maturity, accumulated resets, and frozen mask
        self.utility = {name: torch.zeros_like(param, device=param.device) for name, param in model.named_parameters()}
        self.utility_momentum = {name: torch.zeros_like(param, device=param.device) for name, param in self.utility.items()}
        self.maturity = {name: torch.zeros_like(param, dtype=torch.long, device=param.device) for name, param in self.utility.items()}
        self.accumulated_resets = {name: 0.0 for name in self.utility}
        self.is_frozen = {name: torch.zeros_like(param, dtype=torch.bool, device=param.device) for name, param in model.named_parameters()}

    def update_utility(self, current_step, step_interval=1):
        """Update the utility values for parameters based on their gradients."""
        if current_step % step_interval != 0:
            return
        epsilon = 1e-8
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Compute gradient contribution
                grad_contribution = torch.abs(param.grad) * (torch.abs(param) + epsilon)

                # If optimizer state exists, adjust gradient contribution accordingly
                if self.optimizer is not None:
                    state = self.optimizer.state[param]
                    if 'momentum_buffer' in state:
                        grad_contribution *= (torch.abs(state['momentum_buffer']) + epsilon)
                    if 'exp_avg_sq' in state:
                        grad_contribution /= (torch.sqrt(state.get('exp_avg_sq', torch.zeros_like(param))) + epsilon)

                # Update utility with decay and momentum
                self.utility[name] = self.utility_decay * self.utility[name] + (1 - self.utility_decay) * grad_contribution

    def reset_weights(self, param, indices):
        """Reset selected weights using the specified initialization strategy."""
        if self.reset_init == 'uniform':
            param.view(-1)[indices] = (torch.rand(indices.size(), device=param.device) - 0.5) * 0.1
        elif self.reset_init == 'kaiming':
            reshaped = param.flatten()[indices].view(len(indices), -1)
            init.kaiming_uniform_(reshaped, nonlinearity='relu')
            param.flatten()[indices] = reshaped.flatten()
        elif self.reset_init == 'orthogonal' and len(indices) > 0:
            temp = torch.randn(len(indices), device=param.device)
            init.orthogonal_(temp.view(-1, 1))
            param.flatten()[indices] = temp.flatten()
        elif self.reset_init == 'normal':
            param.view(-1)[indices] = torch.randn(indices.size(), device=param.device) * 0.1
        elif self.reset_init == 'constant':
            param.view(-1)[indices] = 0.0

    def selective_reset(self):
        """Selectively reset inefficient weights and apply freezing mechanism."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                utility_values = self.utility[name].flatten()
                maturity_values = self.maturity[name].flatten()

                # Mark frozen weights based on their utility
                freeze_threshold = torch.quantile(utility_values, 0.9)  # Freeze top 10% by utility
                freeze_mask = utility_values > freeze_threshold
                self.is_frozen[name].view(-1)[freeze_mask] = True

                # Filter unfrozen and mature weights
                valid_mask = (~self.is_frozen[name].view(-1)) & (maturity_values >= self.maturity_threshold)
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]

                # Dynamically determine the number of weights to reset
                num_to_reset = min(len(valid_indices), int(self.reset_rate * len(utility_values)))
                selected_indices = valid_indices[:num_to_reset]

                if len(selected_indices) > 0:
                    self.reset_weights(param, selected_indices)
                    self.utility[name].view(-1)[selected_indices] = 0
                    self.maturity[name].view(-1)[selected_indices] = 0

    def increment_maturity(self):
        """Dynamically increment maturity based on gradient contributions."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_contribution = torch.abs(param.grad) * (torch.abs(param) + 1e-8)
                # Increment maturity for weights with significant gradient contributions
                avg_contribution = grad_contribution.mean()
                std_contribution = grad_contribution.std()
                dynamic_threshold = avg_contribution + std_contribution
                self.maturity[name] += (grad_contribution > dynamic_threshold).long()

    def step(self, current_step, step_interval=1):
        """Perform a single step of the continual backpropagation process."""
        # Update utility values (vectorized operations)
        self.update_utility(current_step, step_interval)

        # Selectively reset weights (dynamic threshold with freezing mechanism)
        self.selective_reset()

        # Dynamically increment maturity
        self.increment_maturity()



# # Example usage
# class SimpleNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.bn = nn.BatchNorm1d(hidden_dim) #Example with BatchNorm
#         self.ln = nn.LayerNorm(hidden_dim) # Example with LayerNorm

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.bn(x)
#         x = self.ln(x)
#         x = self.fc2(x)
#         return x

# # Define model, optimizer, and loss function
# model = SimpleNet(input_dim=10, hidden_dim=20, output_dim=5)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()

# # Instantiate the Continual Backpropagation algorithm
# cbp = ContinualBackpropagation(model)

# # Simulate training
# data = torch.randn(64, 10)
# labels = torch.randint(0, 5, (64,))

# for epoch in range(200):
#     optimizer.zero_grad()
#     outputs = model(data)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     # Perform a Continual Backpropagation step
#     cbp.step()

#     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
