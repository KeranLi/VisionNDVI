import torch
import torch.nn as nn

class InferenceAdapter(nn.Module):
    def __init__(self, input_shape):
        super(InferenceAdapter, self).__init__()
        # Adjust the input dimension to match the flattened input size
        self.fc1 = nn.Linear(input_shape, 128)  # input_shape should match the flattened input size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, input_shape)  # Output dimension is same as input dimension
    
    def forward(self, x):
        # Ensure the input is in float32
        x = x.to(torch.float32)
        
        # Flatten the input
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class FineTuningAdapter(nn.Module):
    def __init__(self, input_size, output_size=512):
        super(FineTuningAdapter, self).__init__()
        # Input size should match the flattened grid size (900 for 30x30 grid)
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        self.fc2 = nn.Linear(output_size, input_size, bias=False)  # Adjust output size to match input size

    def forward(self, x):
        # Ensure the input tensor is on the same device as the model's parameters
        x = x.to(self.fc1.weight.device)  # Ensure input tensor is on the same device as the model's parameters

        # Flatten the input tensor to (batch_size, channels * height * width)
        batch_size, channels, height, width = x.size()
        x = x.reshape(batch_size, -1)  # Use reshape to flatten to a 2D tensor: (batch_size, channels * height * width)

        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Apply second fully connected layer
        
        # Reshape back to the original grid size
        x = x.reshape(batch_size, channels, height, width)  # Reshape back to (batch_size, channels, height, width)
        
        return x
