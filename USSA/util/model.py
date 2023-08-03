import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from util.axial_att import AxialAttention,AxialPositionalEmbedding,AxialImageTransformer

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class Rep1(nn.Module):
    def __init__(self,n_in,r1_hidden_size,dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(n_in,r1_hidden_size)
        # self.act = nn.Sigmoid()
        self.act = nn.SELU()
        # self.act = nn.PReLU()
        # self.act = nn.RReLU()

        self.l2 = nn.Linear(n_in,r1_hidden_size)
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)

    def forward(self,x):
        x = self.dropout(x)
        x1 = self.l1(x) # (b,n,d)
        contro_signal = self.act(x1)
        contro_signal = contro_signal.unsqueeze(2) # (b,n,1,d1)
        x2 = self.l2(x)
        x2 = x2.unsqueeze(1) # (b,1,n,d1)
        y = contro_signal * x2 # (b,n,n,d1)
        return y

class Rep_SELU(nn.Module):
    def __init__(self,n_in,r1_hidden_size,dropout=0.2):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_in,n_in),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_in,n_in//2),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_in//2,r1_hidden_size),
            # nn.SELU()
        )
        for param in self.l1.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

        self.l2 = nn.Linear(n_in,r1_hidden_size)
        torch.nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Linear(n_in,r1_hidden_size, bias=False)
        torch.nn.init.constant_(self.l3.weight, 0)
    
    def forward(self,x):
        contro_signal = self.l1(x)
        contro_signal = contro_signal.unsqueeze(2) # (b,n,1,d1)
        x2 = self.l2(x)
        x2 = x2.unsqueeze(1) # (b,1,n,d1)
        y = contro_signal * x2 # (b,n,n,d1)

        z = self.l3(x)
        z = z.unsqueeze(2) # (b,n,1,d1)
        return y + z

class Rep2(nn.Module):
    def __init__(self,n_in,r2_hidden_size,dropout=0.1):
        super().__init__()
        self.l = nn.Linear(2*n_in,r2_hidden_size)
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.l.weight)

    def forward(self,x):
        b,n,d = x.shape
        x = self.dropout(x)
        x1 = x.unsqueeze(2) # (b,n,1,d)
        x2 = x.unsqueeze(1) # (b,1,n,d)
        x1_ = x1.expand(b,n,n,d)
        x2_ = x2.expand(b,n,n,d)
        x_ = torch.cat((x1_,x2_),dim = -1) # (b,n,n,2d)
        y = self.l(x_) # (b,n,n,d2)

        return y


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """
    def __init__(self, gcn_dim, edge_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)

    def forward(self, weight_prob_softmax, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax += self_loop
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        return weights_gcn_outputs


class BiAxialAttention(nn.Module):
    def __init__(self,num_head,dim):
        super(BiAxialAttention, self).__init__()
        self.attn1 = AxialAttention(
            dim = dim,           # embedding dimension
            dim_index = -1,      # where is the embedding dimension
            heads = num_head,           # number of heads for multi-head attention
            num_dimensions = 2,  # number of axial dimensions (images is 2, video is 3, or more)
        )

        self.attn2 = AxialAttention(
            dim = dim,           # embedding dimension
            dim_index = -1,      # where is the embedding dimension
            heads = num_head,           # number of heads for multi-head attention
            num_dimensions = 2,  # number of axial dimensions (images is 2, video is 3, or more)
        )

    
    def forward(self, x): # x:(b,l,l,h)
        x1,x2 = x, x.permute(0,2,1,3)

        x1 = self.attn1(x1)
        x2 = self.attn2(x2)

        return torch.cat((x, x1, x2),dim=-1)

class BiAxialAttentionTransformer(nn.Module):
    def __init__(self,num_head,dim):
        super(BiAxialAttentionTransformer, self).__init__()
        self.attn1 = AxialImageTransformer(
                dim = 256,
                depth = 2,
                reversible = True
            )

        self.attn2 = AxialImageTransformer(
                dim = 256,
                depth = 2,
                reversible = True
            )
        
        self.conv1x1_1 = nn.Conv2d(768, 256, 1)
        self.conv1x1_2 = nn.Conv2d(768, 256, 1)

    
    def forward(self, x): # x:(b,l,l,h)
        x1,x2 = x, x.permute(0,2,1,3)

        x1 = x1.permute(0,3,1,2) # x:(b,h,l,l)
        x1 = self.attn1(self.conv1x1_1(x1))
        x1 = x1.permute(0,2,3,1)

        x2 = x2.permute(0,3,1,2) # x:(b,h,l,l)
        x2 = self.attn2(self.conv1x1_2(x2))
        x2 = x2.permute(0,2,3,1)

        return torch.cat((x1, x2),dim=-1)


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class Predictor(nn.Module):
    def __init__(self,cls_num,word_rep_size,word_rep_hid_size,in_size,out_size,dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=word_rep_size, n_out=out_size, dropout=dropout)
        self.mlp2 = MLP(n_in=word_rep_size, n_out=out_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=out_size, n_out=cls_num, bias_x=True, bias_y=True)

        self.mlp_rel = nn.Sequential(
            MLP(in_size, word_rep_hid_size, dropout=dropout),
            MLP(word_rep_hid_size, word_rep_hid_size, dropout=dropout)
        )

        self.linear = nn.Linear(word_rep_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self,word_rep,word_pair_rep,is_biaffine=True,td = 0.7):
        a = self.mlp1(word_rep)
        b = self.mlp2(word_rep)
        o1 = self.biaffine(a,b)

        y = self.dropout(self.mlp_rel(word_pair_rep))
        o2 = self.linear(y)

        if not is_biaffine:
            return o2

        return (2 - td) * o1 + td * o2

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers

        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)
        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,
                                     config.conv_hid_size * len(config.dilation), config.ffnn_hid_size,
                                     config.out_dropout)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps) # word_reps: b*l*d

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        return outputs


####################################
### CNN 
####################################

class ConvolutionLayer_pre(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=1, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs

class LayerNormForCNN(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """
        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class ConvolutionLayer2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3, groups=1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, bias=False, padding=kernel_size//2, groups = groups),
                LayerNormForCNN((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()
            ])
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=3//2, groups = groups))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        _x = x  
        for layer in self.cnns:
            if isinstance(layer, LayerNormForCNN):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x)
            else:
                x = layer(x)
        _x = _x.permute(0, 2, 3, 1).contiguous()
        return _x