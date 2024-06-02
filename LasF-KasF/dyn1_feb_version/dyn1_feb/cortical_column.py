

import torch
from torch import nn, Tensor


class Q_group_feb(nn.Module):
    def __init__(self, X_len, X_dim, Y_len, Y_dim, q_dim, p_norm, num_D, H_dim, NL_signal, NLS_use_act, LR_ratio, device):
        super(Q_group_feb, self).__init__()

        self.num_D = num_D
        self.p_norm = p_norm
        self.NL_signal = NL_signal
        self.NLS_use_act = NLS_use_act
        self.LR_ratio = LR_ratio
        self.X_dim = X_dim

        self.X_to_Qins = nn.ModuleList([nn.Linear(X_dim, q_dim) for _ in range(num_D)])
        
        if not self.NL_signal: self.X_to_Sin = nn.Linear(X_dim, Y_dim)
        else:
            self.X_to_Hin = nn.Linear(X_dim, X_dim//self.LR_ratio)
            self.Hin_to_Sin = nn.Linear(X_dim//self.LR_ratio, Y_dim)

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        fixed_coeffs = torch.tensor([(-1)**c_id for c_id in range(num_D)]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.shared_coeffs = fixed_coeffs.to(device)
        #self.shared_coeffs = torch.tensor([(-1)**c_id for c_id in range(num_D)]).unsqueeze(1).unsqueeze(2).unsqueeze(3)
       

    def get_batch_relVecs(self, input_Qs):

        _relVecs = torch.zeros(input_Qs.shape[0], input_Qs.shape[1], input_Qs.shape[2], input_Qs.shape[2], input_Qs.shape[3], device=device)
        for D_id in range(self.num_D): 
            for batch_id in range(input_Qs.shape[1]):
                _relVecs[D_id, batch_id] = input_Qs[D_id, batch_id].unsqueeze(1) - input_Qs[D_id, batch_id]

        return _relVecs


    def forward(self, input_X, src_mask):

        Qins = torch.cat([X_to_Qin(input_X).unsqueeze(0) for X_to_Qin in self.X_to_Qins])
        distMat = torch.cdist(Qins.transpose(1,2), Qins.transpose(1,2), p=self.p_norm)

        relMat = torch.sum(distMat*self.shared_coeffs, 0)

        relMat = relMat/self.X_dim**0.5
        if src_mask != None: relMat = self.softmax(relMat + src_mask)

        if not self.NL_signal: Sin = self.X_to_Sin(input_X)
        else: 
            if self.NLS_use_act: Sin = self.Hin_to_Sin(self.sigmoid(self.X_to_Hin(input_X)))
            else: Sin = self.Hin_to_Sin(self.X_to_Hin(input_X))

        Sout = torch.matmul(relMat, Sin.transpose(0,1))# + Sin.transpose(0,1)

        temp_Qins = Qins.view(Qins.shape[1]*Qins.shape[0],Qins.shape[2],Qins.shape[3])

        temp_Qins = torch.mean(temp_Qins.transpose(0,2), dim=2).transpose(0,1)

        return Sout, temp_Qins


