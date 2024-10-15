import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Encoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)

        x = self.fc(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout):
        super(Decoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.seq_len = seq_len
        
        self.fc = nn.Linear(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        x, _ = self.lstm(x)
        
        return x

class Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout=0.5):
        super(Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = Encoder(input_size=num_feats, 
                               hidden_size=hidden_size, 
                               output_size=latent_seq_len * latent_num_feats, 
                               num_layers=num_layers,
                               dropout=dropout)

        self.decoder = Decoder(input_size=num_feats, 
                               hidden_size=hidden_size, 
                               output_size=latent_seq_len * latent_num_feats, 
                               num_layers=num_layers,
                               seq_len=seq_len,
                               dropout=dropout)                       
    
    def forward(self, x):
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent