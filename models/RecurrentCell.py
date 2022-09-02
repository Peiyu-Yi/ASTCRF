import torch
import torch.nn as nn
from model.AGC import SAGC
import torch.nn.functional as F

class ASTCRFCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(ASTCRFCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out

        self.weight_connect = nn.Parameter(torch.FloatTensor(node_num, node_num))
        self.weight_disconnect = nn.Parameter(torch.FloatTensor(node_num, node_num))

        self.gate = SAGC(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = SAGC(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        x_gconv, A = self.gate(input_and_state, node_embeddings)
        z_r = torch.sigmoid(x_gconv)
        #z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings)[0])
        h = r*state + (1-r)*hc

        #supports = supports[0] + supports[1]
        zeros = torch.zeros_like(A).to(A.device)
        ones = torch.ones_like(A).to(A.device)

        A_connect = torch.where(A > 0, ones, zeros)
        A_disconnect = torch.where(A < 0, ones, zeros)

        degree_connect = self.compute_degrees(A_connect, h, self.weight_connect)
        degree_disconnect = self.compute_degrees(A_disconnect, h, self.weight_disconnect)

        # PEMS04: alpha=0.6 beta=0.2 gamma=0.2
        # PEMS08: alpha=0.6 beta=0.2 gamma=0.2
        # SOLAR: alpha=0.8 beta=0.1 gamma=0.1
        # EXCHANGE: alpha=0.85 beta=0.1 gamma=0.05
        Numerator = 0.8*h - 0.1*torch.bmm(degree_connect, h) + 0.1*torch.bmm(degree_disconnect, h)
        #Denominator = 0.9 - 0.1 * degree_connect + 0.1 * degree_disconnect
        #Denominator = torch.sum(Denominator, dim=-1)
        #B = torch.div(Numerator, Denominator.unsqueeze(2))
        #C = Numerator / Denominator.unsqueeze(2)
        #print(B == C)

        return Numerator

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    def compute_degrees(self, adj, H, weight):
        HHT = torch.bmm(H, H.transpose(1, 2))
        AHHT = torch.mul(adj, HHT)
        AHHTW = torch.einsum('bnm, mk->bnk', AHHT, weight)
        degree_matrix = F.softmax(AHHTW, dim=-1)
        return degree_matrix