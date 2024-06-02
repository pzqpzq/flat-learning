
import math
import torch
from torch import nn, Tensor
from dyn1_feb.neuronal_broadcast import dyn1_block_feb


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DyN_Model(nn.Module):

    def __init__(self, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type='', NoLayers=0, is_Residual=True, \
                    p_norm=0, q_dim=0, num_D=0, num_G=0, dyn1_dropout=0.1, device=''):
        super().__init__()

        self.is_Residual = is_Residual
        self.NoLayers = NoLayers

        self.dyn1_layer = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)
        if self.NoLayers >= 2: self.dyn1_layer2 = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)
        if self.NoLayers >= 3: self.dyn1_layer3 = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)
        if self.NoLayers >= 4: self.dyn1_layer4 = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)  
        if self.NoLayers >= 5: self.dyn1_layer5 = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)
        if self.NoLayers >= 6: self.dyn1_layer6 = dyn1_block_feb(bptt, emsize, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)     



    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:


        output1, global_indicator1 = self.dyn1_layer.forward_vecs(src, src_mask)
        if self.is_Residual: output = output1 + src
        if not self.is_Residual: output = output1
        global_indicator = global_indicator1

        if self.NoLayers >= 2:
            output2, global_indicator2 = self.dyn1_layer2.forward_vecs(output, src_mask)
            if self.is_Residual: output = output2 + output
            if not self.is_Residual: output = output2
            global_indicator += global_indicator2

        if self.NoLayers >= 3:
            output3, global_indicator3 = self.dyn1_layer3.forward_vecs(output, src_mask)
            if self.is_Residual: output = output3 + output
            if not self.is_Residual: output = output3
            global_indicator += global_indicator3

        if self.NoLayers >= 4:
            output4, global_indicator4 = self.dyn1_layer4.forward_vecs(output, src_mask)
            if self.is_Residual: output = output4 + output
            if not self.is_Residual: output = output4
            global_indicator += global_indicator4

        if self.NoLayers >= 5:
            output5, global_indicator5 = self.dyn1_layer5.forward_vecs(output, src_mask)
            if self.is_Residual: output = output5 + output
            if not self.is_Residual: output = output5
            global_indicator += global_indicator5

        if self.NoLayers >= 6:
            output6, global_indicator6 = self.dyn1_layer6.forward_vecs(output, src_mask)
            if self.is_Residual: output = output6 + output
            if not self.is_Residual: output = output6
            global_indicator += global_indicator6


        return output, global_indicator



class LM_Model(nn.Module):

    def __init__(self, bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, p_norm, q_dim, num_D, num_G, ntokens, dyn1_dropout, PE_dropout, device):
        super().__init__()

        self.pos_encoder = PositionalEncoding(emsize, PE_dropout)
       
        self.embedding = nn.Embedding(ntokens, emsize)

        self.d_model = emsize
        self.linear = nn.Linear(emsize, ntokens)

        self.dyn_model_distmap = DyN_Model(bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, \
                                            groupComm_type='distmap-p1p2', NoLayers=4, is_Residual=True, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)

        self.dyn_model_fcn = DyN_Model(bptt, emsize, dyn_Hdim, NL_signal, NLS_use_act, LR_ratio, \
                                            groupComm_type='3layer-FCN', NoLayers=2, is_Residual=False, \
                                            p_norm=p_norm, q_dim=q_dim, num_D=num_D, num_G=num_G, dyn1_dropout=dyn1_dropout, device=device)

        self.init_weights()

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self) -> None:
        initrange = 0.1

        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output_distmap, global_distmap = self.dyn_model_distmap(src, src_mask)
        output_fcn, global_fcn = self.dyn_model_fcn(src, src_mask)

        res_dict = {
            'distmap': {
                'output': self.linear(output_distmap),
                'global_indicator': global_distmap
            },
            'fcn': {
                'output': self.linear(output_fcn),
                'global_indicator': global_fcn
            }
        }
        
        return res_dict

