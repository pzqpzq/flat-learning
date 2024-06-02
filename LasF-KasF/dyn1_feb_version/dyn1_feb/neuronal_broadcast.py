

import torch
from torch import nn, Tensor
from dyn1_feb.cortical_column import Q_group_feb



class dyn1_block_feb(nn.Module):
    def __init__(self, X_len, X_dim, Y_len, Y_dim, H_dim, NL_signal, NLS_use_act, LR_ratio, groupComm_type, p_norm=0, q_dim=0, num_D=0, num_G=0, dyn1_dropout=0.1, device=''):
        super(dyn1_block_feb, self).__init__()

        self.num_G = num_G
        self.dropout = nn.Dropout(p=dyn1_dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()
        self.softmax = nn.Softmax(dim=2)
        self.groupComm_type = groupComm_type

        self.Q_groups = nn.ModuleList()
        for G_id in range(num_G): self.Q_groups.append(Q_group_feb(X_len, X_dim, Y_len, Y_dim, q_dim, p_norm, num_D, H_dim, NL_signal, NLS_use_act, LR_ratio, device))
        self.total_params = sum([sum([param.numel() for name, param in self.Q_groups[G_id].named_parameters()]) for G_id in range(num_G)])

        #self.LSout_to_Y = nn.Linear(H_dim, Y_dim)
        self.LSout_to_Y = nn.Linear(X_dim, Y_dim)

        if self.groupComm_type in ['mlp', 'double-mlp', 'triple-mlp']:
            self.Qstate_layer1 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2 = nn.Linear(round(q_dim*num_G*1.5), num_G)

        if self.groupComm_type in ['double-mlp', 'triple-mlp']:
            self.Qstate_layer1_2 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2_2 = nn.Linear(round(q_dim*num_G*1.5), num_G)

        if self.groupComm_type in ['triple-mlp']:
            self.Qstate_layer1_3 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2_3 = nn.Linear(round(q_dim*num_G*1.5), num_G)

        if self.groupComm_type in ['2layer-FCN']:
            self.Qstate_layer1 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2 = nn.Linear(round(q_dim*num_G*1.5), num_G)

        if self.groupComm_type in ['3layer-FCN']:
            self.Qstate_layer1 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2 = nn.Linear(round(q_dim*num_G*1.5), round(q_dim*num_G*1.5))
            self.Qstate_layer3 = nn.Linear(round(q_dim*num_G*1.5), num_G)

        if self.groupComm_type in ['4layer-FCN']:
            self.Qstate_layer1 = nn.Linear(q_dim*num_G, round(q_dim*num_G*1.5))
            self.Qstate_layer2 = nn.Linear(round(q_dim*num_G*1.5), round(q_dim*num_G*1.5))
            self.Qstate_layer3 = nn.Linear(round(q_dim*num_G*1.5), round(q_dim*num_G*1.5))
            self.Qstate_layer4 = nn.Linear(round(q_dim*num_G*1.5), num_G)


    def forward_vecs(self, input_X, src_mask=None):

        if self.groupComm_type == 'None':
            S_out = self.Q_groups[0](input_X, src_mask)[0]
            for G_id in range(1, self.num_G): S_out = S_out + self.Q_groups[G_id](input_X, src_mask)[0]
            cur_Y = self.LSout_to_Y(self.sigmoid(S_out)).transpose(0,1)
            return self.dropout(cur_Y)

        Sout_list, Qstate_list = [], []
        
        for G_id in range(self.num_G):
            Sout, Qstate = self.Q_groups[G_id](input_X, src_mask)
            Sout_list.append(Sout)
            Qstate_list.append(Qstate)
            
        if self.groupComm_type in ['mlp', 'double-mlp', 'triple-mlp', '2layer-FCN', '3layer-FCN', '4layer-FCN']:
            Qstate_vecs = torch.cat(Qstate_list, dim=1)
            Qstate_tensor = torch.stack(Qstate_list).transpose(0,1)
            Qstate_map = self.softmax(torch.cdist(Qstate_tensor, Qstate_tensor, p=1))
            global_indicator = 1/(torch.mean(Qstate_map)+0.1)

        if self.groupComm_type in ['mlp', 'double-mlp', 'triple-mlp']:
            Qstate_weights = self.Qstate_layer2(self.relu(self.Qstate_layer1(Qstate_vecs)))

        if self.groupComm_type in ['double-mlp', 'triple-mlp']:
            Qstate_weights_2 = self.Qstate_layer2_2(self.relu(self.Qstate_layer1_2(Qstate_vecs)))

        if self.groupComm_type in ['triple-mlp']:
            Qstate_weights_3 = self.Qstate_layer2_3(self.relu(self.Qstate_layer1_3(Qstate_vecs)))

        if self.groupComm_type in ['2layer-FCN']:
            Qstate_weights = self.Qstate_layer2(self.relu(self.Qstate_layer1(Qstate_vecs)))

        if self.groupComm_type in ['3layer-FCN']:
            Qstate_weights = self.Qstate_layer3(self.relu(self.Qstate_layer2(self.relu(self.Qstate_layer1(Qstate_vecs)))))        

        if self.groupComm_type in ['4layer-FCN']:
            Qstate_weights = self.Qstate_layer4(self.relu(self.Qstate_layer3(self.relu(self.Qstate_layer2(self.relu(self.Qstate_layer1(Qstate_vecs)))))))      

        if self.groupComm_type in ['mlp', 'double-mlp', 'triple-mlp', '2layer-FCN', '3layer-FCN', '4layer-FCN']:
            temp_Y = Sout_list[0]*Qstate_weights[:,0].unsqueeze(1).unsqueeze(2)
            for G_id in range(1, self.num_G): temp_Y += Sout_list[G_id]*Qstate_weights[:,G_id].unsqueeze(1).unsqueeze(2)

        if self.groupComm_type in ['double-mlp', 'triple-mlp']:
            temp_Y_2 = Sout_list[0]*Qstate_weights_2[:,0].unsqueeze(1).unsqueeze(2)
            for G_id in range(1, self.num_G): temp_Y_2 += Sout_list[G_id]*Qstate_weights_2[:,G_id].unsqueeze(1).unsqueeze(2)

        if self.groupComm_type in ['triple-mlp']:
            temp_Y_3 = Sout_list[0]*Qstate_weights_3[:,0].unsqueeze(1).unsqueeze(2)
            for G_id in range(1, self.num_G): temp_Y_3 += Sout_list[G_id]*Qstate_weights_3[:,G_id].unsqueeze(1).unsqueeze(2)

        if self.groupComm_type in ['distmap-p1', 'distmap-p1p2', 'distmap-p1p2p3']:
            Qstate_tensor = torch.stack(Qstate_list).transpose(0,1)
            Qstate_map = self.softmax(1/(1+torch.cdist(Qstate_tensor, Qstate_tensor, p=1)))

            Sout_tensor = torch.stack(Sout_list).transpose(0,1)

            global_indicator = 1/(torch.mean(Qstate_map)+0.1)

            temp_Y = torch.sum(Qstate_map[:,:,0].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)
            for G_id in range(1, self.num_G): temp_Y += torch.sum(Qstate_map[:,:,G_id].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)

        if self.groupComm_type in ['distmap-p2']:
            Qstate_tensor = torch.stack(Qstate_list).transpose(0,1)
            Qstate_map = self.softmax(1/(1+torch.cdist(Qstate_tensor, Qstate_tensor, p=2)))

            Sout_tensor = torch.stack(Sout_list).transpose(0,1)

            global_indicator = 1/(torch.mean(Qstate_map)+0.1)

            temp_Y_2 = torch.sum(Qstate_map[:,:,0].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)
            for G_id in range(1, self.num_G): temp_Y_2 += torch.sum(Qstate_map[:,:,G_id].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)


        if self.groupComm_type in ['distmap-p1p2', 'distmap-p1p2p3']:
            Qstate_map_2 = self.softmax(1/(1+torch.cdist(Qstate_tensor, Qstate_tensor, p=2)))

            temp_Y_2 = torch.sum(Qstate_map_2[:,:,0].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)
            for G_id in range(1, self.num_G): temp_Y_2 += torch.sum(Qstate_map_2[:,:,G_id].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)

        if self.groupComm_type in ['distmap-p1p2p3']:
            Qstate_map_3 = self.softmax(1/(1+torch.cdist(Qstate_tensor, Qstate_tensor, p=3)))

            temp_Y_3 = torch.sum(Qstate_map_3[:,:,0].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)
            for G_id in range(1, self.num_G): temp_Y_3 += torch.sum(Qstate_map_3[:,:,G_id].unsqueeze(2).unsqueeze(3)*Sout_tensor, dim=1)

        if self.groupComm_type in ['mlp', '2layer-FCN', '3layer-FCN', '4layer-FCN', 'distmap-p1']: cur_Y = self.LSout_to_Y(self.sigmoid(temp_Y)).transpose(0,1)
        if self.groupComm_type in ['double-mlp', 'distmap-p1p2']: cur_Y = self.LSout_to_Y(self.sigmoid(temp_Y)+self.sigmoid(temp_Y_2)).transpose(0,1)
        if self.groupComm_type in ['triple-mlp', 'distmap-p1p2p3']: cur_Y = self.LSout_to_Y(self.sigmoid(temp_Y)+self.sigmoid(temp_Y_2)+self.sigmoid(temp_Y_3)).transpose(0,1)        
        if self.groupComm_type in ['distmap-p2']: cur_Y = self.LSout_to_Y(self.sigmoid(temp_Y_2)).transpose(0,1)

        return self.dropout(cur_Y), global_indicator#*Gind_ratio


