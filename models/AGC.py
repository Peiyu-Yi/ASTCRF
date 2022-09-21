import torch
import torch.nn.functional as F
import torch.nn as nn


class SAGC(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(SAGC, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k, dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.linear1 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        node_embeddings = self.linear1(node_embeddings)
        A = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        supports = F.softmax(F.relu(A), dim=1)

        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias     #b, N, dim_out
        return x_gconv, A