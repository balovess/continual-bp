import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class ContinualBackpropagation:
    def __init__(self, model, reset_rate=0.001, maturity_threshold=100, utility_decay=0.9, reset_init='uniform'):
        """
        Initializes the Continual Backpropagation algorithm.

        Args:
            model (nn.Module): The neural network model.
            reset_rate (float): The proportion of mature units to reset in each layer (rho).
            maturity_threshold (int): The number of updates before a new unit is eligible for resetting (m).
            utility_decay (float): The decay factor for the utility values (eta).
            reset_init (str): The initialization method for reset weights ('uniform' or 'kaiming').
        """
        self.model = model
        self.reset_rate = reset_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.reset_init = reset_init

        # Initialize utility and maturity trackers for weights only
        self.utility = {name: torch.zeros_like(param) for name, param in model.named_parameters() if 'weight' in name}
        self.maturity = {name: torch.zeros_like(param, dtype=torch.long) for name, param in model.named_parameters() if 'weight' in name}
        self.accumulated_resets = {name: torch.tensor(0.0) for name in self.utility}

    def update_utility(self):
        """
        Updates the utility values based on the current gradients and weights.
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name and hasattr(param, 'grad') and param.grad is not None:  # Ensure gradient exists
                contribution = torch.abs(param.grad) * torch.abs(param)  # Element-wise utility calculation
                self.utility[name] = self.utility_decay * self.utility[name] + (1 - self.utility_decay) * contribution

    def selective_reset(self):
        """
        Selectively resets units with low utility and high maturity.
        This function now robustly handles various parameter types (bias, weight, running_mean, running_var)
        in BatchNorm, LayerNorm, and other modules, preventing IndexError.
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                utility_values = self.utility[name].flatten()
                num_to_reset_float = self.reset_rate * len(utility_values)
                self.accumulated_resets[name] += num_to_reset_float
                num_to_reset = int(self.accumulated_resets[name].item())
                self.accumulated_resets[name] -= num_to_reset

                if num_to_reset > 0:
                    _, indices = torch.topk(utility_values, num_to_reset, largest=False)
                    maturity_mask = self.maturity[name].flatten() >= self.maturity_threshold
                    valid_indices = torch.masked_select(indices, maturity_mask[indices])

                    if len(valid_indices) > 0:
                        print(f"Resetting {name} with {len(valid_indices)} neurons")
                        new_param = param.detach().clone()
                        if self.reset_init == 'uniform':
                            new_param.flatten()[valid_indices] = (torch.rand_like(new_param.flatten()[valid_indices]) - 0.5) * 0.02
                        elif self.reset_init == 'kaiming':
                            init.kaiming_uniform_(new_param.flatten()[valid_indices].view(len(valid_indices), -1), nonlinearity='relu')
                        param.data.copy_(new_param)

                        self.utility[name].flatten()[valid_indices] = 0
                        self.maturity[name].flatten()[valid_indices] = 0

                        module_name = name.replace(".weight", "")
                        module = self.model.get_submodule(module_name)

                        # Generic handling of all possible parameters: bias, weight, running_mean, running_var
                        param_names_to_reset = ["bias", "weight", "running_mean", "running_var"]
                        for param_name in param_names_to_reset:
                            if hasattr(module, param_name) and getattr(module, param_name) is not None:
                                param_to_reset = getattr(module, param_name)
                                num_features = param_to_reset.shape[0]
                                valid_indices_param = valid_indices[valid_indices < num_features]  # Key: Filter indices
                                if len(valid_indices_param) > 0:  # Prevent indexing empty lists
                                    param_to_reset.data[valid_indices_param] = (
                                        0.0 if param_name in ["bias", "running_mean"] else
                                        1.0 if param_name in ["weight", "running_var"] else
                                        None
                                    )

    def increment_maturity(self):
        """
        Increments the maturity of all units.
        """
        for name in self.maturity:
            self.maturity[name] += 1

    def step(self):
        """
        Performs a single update step: update utility, selective reset, and increment maturity.
        """
        self.update_utility()
        self.selective_reset()
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
