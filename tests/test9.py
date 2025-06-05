import torch
import torch.nn as nn

# Create a network with recurrent and transformer layers
class RecurrentTransformer(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=2, num_heads=4):
        super().__init__()
        # Recurrent layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Transformer layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4
            ),
            num_layers=2
        )
        
        # Output layers
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Transformer forward pass
        transformer_out = self.transformer_encoder(x)
        
        # Combine outputs
        combined = lstm_out + gru_out + rnn_out + transformer_out
        
        # Final output
        output = self.fc(combined)
        return output

# Create input data
batch_size = 2
seq_length = 5
input_size = 10
input_data = torch.randn(batch_size, seq_length, input_size)
print("Input shape:", input_data.shape)

# Initialize model
model = RecurrentTransformer()
print("\nModel architecture:")
print(model)

# Forward pass
output = model(input_data)
print("\nOutput shape:", output.shape)

# Different recurrent cell types
print("\nDifferent recurrent cell types:")
lstm_cell = nn.LSTMCell(input_size, hidden_size=20)
gru_cell = nn.GRUCell(input_size, hidden_size=20)
rnn_cell = nn.RNNCell(input_size, hidden_size=20)

# Process sequence with cells
h_lstm = torch.zeros(batch_size, 20)
c_lstm = torch.zeros(batch_size, 20)
h_gru = torch.zeros(batch_size, 20)
h_rnn = torch.zeros(batch_size, 20)

print("\nProcessing sequence with cells:")
for t in range(seq_length):
    h_lstm, c_lstm = lstm_cell(input_data[:, t, :], (h_lstm, c_lstm))
    h_gru = gru_cell(input_data[:, t, :], h_gru)
    h_rnn = rnn_cell(input_data[:, t, :], h_rnn)
    print(f"Step {t} - LSTM hidden state shape:", h_lstm.shape)

# Attention mechanisms
print("\nAttention mechanisms:")
# Self-attention
self_attention = nn.MultiheadAttention(embed_dim=20, num_heads=4)
attn_output, attn_weights = self_attention(
    input_data.transpose(0, 1),
    input_data.transpose(0, 1),
    input_data.transpose(0, 1)
)
print("Self-attention output shape:", attn_output.shape)
print("Attention weights shape:", attn_weights.shape)

# Positional encoding
print("\nPositional encoding:")
pos_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=20, nhead=4),
    num_layers=1
)
pos_output = pos_encoder(input_data)
print("Positional encoding output shape:", pos_output.shape) 