from torch import nn
from torch.nn import functional as F

class BatchedLSTM(nn.Module):
    """
    A batched LSTM-based decision module for agents.
    Handles sequence processing for multiple agents in parallel,
    projecting input features through an LSTM and a final linear layer to output action probabilities.

    Args:
        input_size (int): Number of input features per agent.
        hidden_size (int): Size of LSTM hidden state.
        output_size (int): Number of output actions/classes.
        num_layers (int): Number of stacked LSTM layers.
        device (str): Target device ('cpu' or 'cuda').

    Forward input:
        x (torch.Tensor): [batch, seq=1, input_size] - Input features.
        h (torch.Tensor): [num_layers, batch, hidden_size] - Previous hidden state.
        c (torch.Tensor): [num_layers, batch, hidden_size] - Previous cell state.

    Forward output:
        probs (torch.Tensor): [batch, output_size] - Probability distribution over actions for each agent.
        h_new (torch.Tensor): [num_layers, batch, hidden_size] - Updated hidden state.
        c_new (torch.Tensor): [num_layers, batch, hidden_size] - Updated cell state.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device="cpu"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
        self.to(self.device)

    def forward(self, x, h, c):
        """
        Runs a forward pass of the batched LSTM network.
        Args:
            x (torch.Tensor): [batch, seq=1, input_size] - Input feature tensor.
            h (torch.Tensor): [num_layers, batch, hidden_size] - Previous hidden state.
            c (torch.Tensor): [num_layers, batch, hidden_size] - Previous cell state.
        Returns:
            probs (torch.Tensor): [batch, output_size] - Output action probabilities.
            h_new (torch.Tensor): [num_layers, batch, hidden_size] - Updated hidden state.
            c_new (torch.Tensor): [num_layers, batch, hidden_size] - Updated cell state.
        """
        # LSTM expects input of shape [batch, seq, input_size]
        lstm_out, (h_new, c_new) = self.lstm(x, (h, c))  # lstm_out: [batch, seq=1, hidden_size]
        out = self.fc(lstm_out.squeeze(1))                # [batch, output_size]
        probs = F.softmax(out, dim=-1)                    # [batch, output_size]
        return probs, h_new, c_new
