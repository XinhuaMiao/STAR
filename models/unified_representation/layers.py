import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------- Temporal Adaptive Normalization (TAN) --------------------
class TAN(nn.Module):
    """ Multi-Scale Adaptive Normalization """
    def __init__(self, num_features, num_scales=3, eps=1e-5): # Number of Scales Ns = 3
        super(TAN, self).__init__()
        self.eps = eps
        self.num_scales = num_scales
        
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_features, num_features, kernel_size=2**i, padding=2**(i-1)),
                nn.AdaptiveAvgPool1d(1)
            ) for i in range(1, num_scales+1)
        ])
        
        self.param_generator = nn.Sequential(
            nn.Linear(num_scales * num_features, 4 * num_features),
            nn.LeakyReLU(),
            nn.Linear(4 * num_features, 4 * num_features)
        )
        
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            # Multi-Scale Features Extraction
            scale_features = []
            x_t = x.transpose(1, 2)  # (batch, channels, seq_len)
            
            for conv in self.scale_convs:
                scale_feat = conv(x_t)  # (batch, channels, 1)
                scale_features.append(scale_feat.squeeze(-1))
            
            context = torch.cat(scale_features, dim=-1)  # (batch, channels*num_scales)
            
            params = self.param_generator(context)  # (batch, 4*channels)
            gamma1, delta1, gamma2, delta2 = params.chunk(4, dim=-1)
            
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True) + self.eps
            
            # Normalization
            x_norm_base = (x - mu) / sigma
            
            # Adjustment
            x_norm_adapt = x_norm_base * gamma1.unsqueeze(1) + delta1.unsqueeze(1)
            x_norm = self.alpha * x_norm_base + (1 - self.alpha) * x_norm_adapt + self.beta
            
            self.mu = mu
            self.sigma = sigma
            self.gamma1 = gamma1
            self.delta1 = delta1
            self.gamma2 = gamma2
            self.delta2 = delta2
            
            return x_norm
        
        else:  
            # De-normalization
            term1 = (x - self.beta) / self.alpha
            term2 = (term1 - self.delta1.unsqueeze(1)) / (self.gamma1.unsqueeze(1) + self.eps)
            
            x_denorm = term2 * self.sigma + self.mu
            x_denorm = x_denorm * self.gamma2.unsqueeze(1) + self.delta2.unsqueeze(1)
            
            return x_denorm

class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers, use_tan=True):
        super(TransformerEncoder, self).__init__()
        self.use_tan= use_tan
        
        if use_tan:
            self.tan = TAN(in_dim)
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim*4,  # FFN
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, features):

        if self.use_tan:
            features = self.tan(features, mode='norm')
        
        h = self.transformer_encoder(features)
        h = F.leaky_relu(h)

        if self.use_tan:
            h = self.tan(h, mode='denorm')
            
        return h

# -------------------- Spatial Adaptive Normalization (SAN) --------------------
class SAN(nn.Module):
    """ Graph-Structure Adaptive Normalization """
    def __init__(self, num_features, eps=1e-5):
        super(SAN, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.degree_embedding = nn.Embedding(100, num_features)  # Maximum Degree D = 100
        
    def forward(self, x, g):
        # Degree Embedding
        degrees = g.in_degrees().clamp(0, 99)
        deg_feat = self.degree_embedding(degrees)
        
        deg_weight = deg_feat.mean(dim=0, keepdim=True)
        deg_bias = deg_feat.std(dim=0, keepdim=True)
        
        # Normalization
        mu = x.mean(dim=0, keepdim=True)
        sigma = x.std(dim=0, keepdim=True) + self.eps
        x_norm = (x - mu) / sigma
        
        return x_norm * (self.alpha + deg_weight) + (self.beta + deg_bias)
    
    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
        # Input Projection
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim if i > 0 else hidden_dim
            output_dim = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(dgl.nn.GraphConv(input_dim, output_dim, norm=norm))
        
        self.san = SAN(out_dim)
        
        # Residual Projection
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, g, features):

        residual = self.residual_proj(features)
        h = self.input_proj(features)
        
        for i, conv in enumerate(self.convs):
            h_res = h if i > 0 else None  # Without Residual for i = 0
            
            h = conv(g, h)
            
            # Residual
            if h_res is not None and h.shape == h_res.shape:
                h = h + h_res
            
            if i < self.num_layers - 1:
                h = F.leaky_relu(h)
                h = self.dropout(h)
        
        # Normalization
        h = self.san(h, g)
        
        # Residual Connection
        if h.shape == residual.shape:
            h = h + residual
        
        return h

class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(Extractor, self).__init__()
        self.TFEncoder = TransformerEncoder(tf_in_dim, num_heads, tf_layers)
        self.GRUEncoder = nn.GRU(gnn_in_dim, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.GraphEncoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, norm='none')
        
    def forward(self, g, features):
        bacth_size, series_len, instance_num, channel_dim = features.shape # 2,5,46,130
        h = features.permute(0,1,3,2)
        h = h.view(-1, channel_dim, instance_num)
        h = self.TFEncoder(h)
        h = h.permute(0,2,1).view(bacth_size, series_len, instance_num, channel_dim).permute(0,2,1,3).reshape(-1, series_len, channel_dim) # 92,5,130
        output, h_n = self.GRUEncoder(h)
        h = F.leaky_relu(h_n[-1]) # 92,32
        h = self.GraphEncoder(g, h) # 92, 32
        h = h.view(bacth_size, instance_num, -1) # 2,46,32
        return h

class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        h = F.leaky_relu(self.mlp(features))
        return h

class AutoRegressor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(AutoRegressor, self).__init__()
        self.extractor = Extractor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        self.regressor = Regressor(gru_hidden_dim, gnn_in_dim)
        
    def forward(self, g, features):
        z = self.extractor(g, features)
        h = self.regressor(z)
        return z, h

# end Neural Network Architecture -----------------------------------------



# -------------------- Collate Function  -------------------------
# Creating a DataLoader for auto_regressor.
def collate_AR(samples):
    timestamps, graphs, feats, targets = map(list, zip(*samples))
    batched_ts = torch.stack(timestamps)
    batched_graphs = dgl.batch(graphs)
    batched_feats = torch.stack(feats)
    batched_targets = torch.stack(targets)
    return  batched_ts, batched_graphs, batched_feats, batched_targets

def create_dataloader_AR(samples, window_size=6, max_gap=60, batch_size=2, shuffle=False):
    # sliding_time_windows
    series_samples = [samples[i:i+window_size] for i in range(len(samples) - window_size + 1)]
    series_samples = [
        series_sample for series_sample in series_samples
            if all(abs(series_sample[i][0] - series_sample[i+1][0]) <= max_gap 
                for i in range(len(series_sample) - 1))
    ]
    # create a dataloader
    dataset = [[
            torch.tensor(series_sample[-1][0]),
            series_sample[-1][1], 
            torch.stack([step[2] for step in series_sample[:-1]]),
            torch.tensor(series_sample[-1][2])
        ] for _, series_sample in enumerate(series_samples)]
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_AR)
    return dataloader

# end Collate Function  ------------------------------------------