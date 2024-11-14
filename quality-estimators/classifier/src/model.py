import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the Multi-Head Attention module.

        :param d_model: Dimension of the input and output features.
        :param num_heads: Number of attention heads.
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

class Encoder(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        """
        Transformer Encoder module with multi-head attention and feedforward layer.
        
        :param in_size: Dimension of the input features.
        :param out_size: Dimensionality of the output features.
        :param num_heads: Number of attention heads for multi-head attention.
        :param dropout: Dropout rate for regularization.
        """
        super(Encoder, self).__init__()
        
        self.attn_layer = MultiHeadAttention(in_size, num_heads)
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for Transformer Encoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, in_size).
        :return: Tuple containing:
            - Encoded tensor of shape (batch_size, seq_len, out_size).
            - Attention matrix of shape (batch_size, seq_length, in_size).
        """
        attn_matrix = self.attn_layer(Q=x, K=x, V=x)
        x = x + attn_matrix

        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        
        return x, attn_matrix

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        """
        Transformer Decoder module with multi-head attention and feedforward layer.
        
        :param in_size: Dimension of the input features.
        :param out_size: Dimensionality of the output features.
        :param num_heads: Number of attention heads for multi-head attention.
        :param dropout: Dropout rate for regularization.
        """
        super(Decoder, self).__init__()
        
        self.linear = nn.Linear(in_size, out_size)
        self.attn_layer = MultiHeadAttention(out_size, num_heads)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for Transformer Decoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, hidden_dim).
        :return: Tuple containing:
            - Decoded tensor of shape (batch_size, seq_len, out_size).
            - Attention matrix of shape (batch_size, seq_length, out_size).
        """
        x = self.linear(x)

        attn_matrix = self.attn_layer(Q=x, K=x, V=x)
        x = x + attn_matrix
        
        x = self.dropout(x)
        x = self.relu(x)
        
        return x, attn_matrix

class Transformer(nn.Module):
    def __init__(self, in_size=3, hidden_dim=4, out_size=5, num_heads=1, dropout=0.5):
        """
        Transformer model for classification, combining an encoder with multi-head attention 
        and a classifier. The encoder captures complex dependencies in the input data, and 
        the classifier outputs the logits for classification.

        :param in_size: Size of the input features.
        :param hidden_dim: Dimensionality of the model's internal representation.
        :param out_size: Size of the output classes.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout rate for regularization.
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(in_size, hidden_dim, num_heads, dropout)
        self.decoder = Decoder(hidden_dim, in_size, num_heads, dropout)
        self.classifier = nn.Linear(in_size, out_size)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights and biases of the classifier linear layer:
        - Set the bias of the classifier linear layer to zero.
        - Initialize the weights with values drawn from a Xavier uniform distribution.
        """
        self.classifier.bias.data.zero_()
        nn.init.xavier_uniform_(self.classifier.weight.data)             
    
    def forward(self, x):
        """
        Forward pass for Transformer.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Tuple containing:
            - Logits of shape (batch_size, seq_len, out_size) for classification.
            - Attention matrix tensor of shape (batch_size, seq_length, in_size).
        """
        enc_x, enc_attn_matrix = self.encoder(x)
        dec_x, dec_attn_matrix = self.decoder(enc_x)

        logits = self.classifier(dec_x)

        attn_matrix = (enc_attn_matrix + dec_attn_matrix) / 2
        
        return logits, attn_matrix