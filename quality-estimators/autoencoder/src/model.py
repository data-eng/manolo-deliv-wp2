import math
import torch
import torch.nn as nn

class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        LSTM Encoder module.
        
        :param input_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param output_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Encoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for LSTM Encoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, input_size).
        :return: Encoded output tensor of shape (batch_size, output_size).
        """
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)

        x = self.fc(x)
        
        return x

class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout):
        """
        LSTM Decoder module.
        
        :param input_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param output_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param seq_len: Length of the output sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Decoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.seq_len = seq_len
        
        self.fc = nn.Linear(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for LSTM Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, output_size).
        :return: Decoded output tensor of shape (batch_size, seq_len, input_size).
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        x, _ = self.lstm(x)
        
        return x

class LSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout=0.5):
        """
        LSTM-based Autoencoder module combining an encoder and a decoder. This module uses LSTM layers 
        to capture temporal dependencies in sequential data and reduces the sequence length by taking 
        the last element of the sequence.
        
        :param seq_len: Length of the input sequence.
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param hidden_size: Number of hidden units in the LSTM.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = LSTM_Encoder(input_size=num_feats, 
                                    hidden_size=hidden_size, 
                                    output_size=latent_seq_len * latent_num_feats, 
                                    num_layers=num_layers,
                                    dropout=dropout)

        self.decoder = LSTM_Decoder(input_size=num_feats, 
                                    hidden_size=hidden_size, 
                                    output_size=latent_seq_len * latent_num_feats, 
                                    num_layers=num_layers,
                                    seq_len=seq_len,
                                    dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for LSTM Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output and latent representation.
        """
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent
    
class ConvLSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout):
        """
        ConvLSTM Encoder module.
        
        :param input_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param output_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param seq_len: Length of the input sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Encoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.conv = nn.Conv1d(hidden_size, output_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for ConvLSTM Encoder.

        :param x: Input tensor of shape (batch_size, seq_len, input_size).
        :return: Encoded output tensor of shape (batch_size, output_size).
        """
        x, _ = self.lstm(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(2)
        
        return x

class ConvLSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout):
        """
        ConvLSTM Decoder module.
        
        :param input_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param output_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param seq_len: Length of the output sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Decoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.conv_transpose = nn.ConvTranspose1d(in_channels=output_size, out_channels=hidden_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for the ConvLSTM Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, output_size).
        :return: Decoded output tensor of shape (batch_size, seq_len, input_size).
        """
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_transpose(x)
        x = x.transpose(1, 2)

        x = self.dropout(x)

        x, _ = self.lstm(x)
        
        return x

class ConvLSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout=0.5):
        """
        ConvLSTM-based Autoencoder module combining an encoder and a decoder. This module integrates 
        convolutional layers with LSTMs to effectively process spatiotemporal data and uses 1D 
        convolutional layers to reduce the sequence length.
        
        :param seq_len: Length of the input sequence.
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param hidden_size: Number of hidden units in the LSTM.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = ConvLSTM_Encoder(input_size=num_feats, 
                                        hidden_size=hidden_size, 
                                        output_size=latent_seq_len * latent_num_feats, 
                                        num_layers=num_layers,
                                        seq_len=seq_len,
                                        dropout=dropout)

        self.decoder = ConvLSTM_Decoder(input_size=num_feats, 
                                        hidden_size=hidden_size, 
                                        output_size=latent_seq_len * latent_num_feats, 
                                        num_layers=num_layers,
                                        seq_len=seq_len,
                                        dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for the ConvLSTM Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output and latent representation.
        """
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the Multi-Head Attention module. The module computes attention weights with shape 
        torch.Size([batch_size, num_heads, seq_length, seq_length]), which can be accessed as 
        model.encoder[layer_id].self_attn.attn_weights.

        :param d_model: dimension of the input and output features
        :param num_heads: number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_weights = None
        
    def attention_scores(self, Q, K, V):
        """
        Calculate the attention scores and apply them to the values.
        
        :param Q: Query matrix.
        :param K: Key matrix.
        :param V: Value matrix.
        :return: Attention scores per head.
        """
        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        scaling_factor = math.sqrt(self.d_k)
        attn_scores = dot_product / scaling_factor

        attn_probs = torch.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_probs

        attn_scores = torch.matmul(attn_probs, V)

        return attn_scores
        
    def split_heads(self, x):
        """
        Split the input into multiple heads.

        :param x: Tensor (batch_size, seq_length, d_model).
        :return: Tensor (batch_size, num_heads, seq_length, d_k).
        """
        batch_size, seq_length, _ = x.size()

        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        
        return x
        
    def combine_heads(self, x):
        """
        Combine multiple heads into a single tensor.

        :param x: Tensor (batch_size, num_heads, seq_length, d_k).
        :return: Tensor (batch_size, seq_length, d_model).
        """
        batch_size, _, seq_length, _ = x.size()

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_length, self.d_model)
        
        return x
        
    def forward(self, Q, K, V):
        """
        Forward pass for multi-head attention.

        :param Q: Query matrix.
        :param K: Key matrix.
        :param V: Value matrix.
        :return: Multi-head attention matrix.
        """
        Q = self.split_heads(x=self.W_q(Q))
        K = self.split_heads(x=self.W_k(K))
        V = self.split_heads(x=self.W_v(V))
        
        attn_scores = self.attention_scores(Q, K, V)
        attn_matrix = self.combine_heads(attn_scores)
        
        output = self.W_o(attn_matrix)

        return output

class Attn_Encoder(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, seq_len, dropout):
        """
        Multi-Head Attention Encoder module.
        
        :param input_size: Size of the input features.
        :param output_size: Size of the output feature vector.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param seq_len: Length of the input sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Encoder, self).__init__()
        
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=input_size, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.conv = nn.Conv1d(input_size, output_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for Attention Encoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, input_size).
        :return: Encoded output tensor of shape (batch_size, output_size).
        """
        for attn_layer in self.attn_layers:
            x = attn_layer(Q=x, K=x, V=x)

        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(2)
        
        return x

class Attn_Decoder(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, seq_len, dropout):
        """
        Multi-Head Attention Decoder module.
        
        :param input_size: Size of the input features.
        :param output_size: Size of the output feature vector.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param seq_len: Length of the output sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Decoder, self).__init__()
        
        self.seq_len = seq_len

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=input_size, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.conv_transpose = nn.ConvTranspose1d(in_channels=output_size, out_channels=input_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for Attention Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, output_size).
        :return: Decoded output tensor of shape (batch_size, seq_len, input_size).
        """
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_transpose(x)
        x = x.transpose(1, 2)

        x = self.dropout(x)

        for attn_layer in self.attn_layers:
            x = attn_layer(Q=x, K=x, V=x)
        
        return x

class Attn_Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, num_heads, num_layers, dropout=0.5):
        """
        Multi-Head Attention-based Autoencoder module combining an encoder and a decoder. This module 
        uses multi-head attention mechanisms to capture complex dependencies in the input data, and 
        applies 1D convolutional layers to reduce the sequence length.
        
        :param seq_len: Length of the input sequence.
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = Attn_Encoder(input_size=num_feats,
                                    output_size=latent_seq_len * latent_num_feats,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    seq_len=seq_len,
                                    dropout=dropout)

        self.decoder = Attn_Decoder(input_size=num_feats,
                                    output_size=latent_seq_len * latent_num_feats,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    seq_len=seq_len,
                                    dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for Attention Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output and latent representation.
        """
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent