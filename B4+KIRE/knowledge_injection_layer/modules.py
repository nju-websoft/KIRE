from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Combineloss(nn.Module):
    def __init__(self, config):
        super(Combineloss, self).__init__()
        self.config = config
        self.small_inf = 1e-30
        if self.config.loss_combine != 'linear':  # https://www.zhihu.com/question/375794498
            self.alpha_re = nn.Parameter(torch.ones(1).float(), True)
            if self.config.add_coref_flag:
                self.alpha_coref = nn.Parameter(torch.ones(1).float(), True)
            if self.config.add_kg_flag:
                self.alpha_kg = nn.Parameter(torch.ones(1).float(), True)

    def forward(self, re_loss, coref_loss, kg_loss):
        if self.config.loss_combine == 'linear':
            loss = self.config.alpha_re * re_loss
            if coref_loss is not None:
                loss += self.config.alpha_coref * coref_loss

            if kg_loss is not None:
                loss += self.config.alpha_kg * kg_loss

        else:
            alpha_re = F.relu(self.alpha_re)
            loss = (1 / (alpha_re**2 + self.small_inf)) * re_loss + 2 * torch.log(alpha_re + self.small_inf)
            if coref_loss is not None:
                alpha_coref = F.relu(self.alpha_coref)
                loss += ( 1 / (alpha_coref**2 + self.small_inf)) * coref_loss + 2 * torch.log(alpha_coref + self.small_inf)

            if kg_loss is not None:
                alpha_kg = F.relu(self.alpha_kg)
                loss += ( 1 / (alpha_kg**2 + self.small_inf)) * kg_loss + 2 * torch.log(alpha_kg + self.small_inf)

        return loss


class GCN(nn.Module):

    def __init__(self, in_dim, mem_dim, num_layers, gpu, gcn_in_drop=0.5, gcn_out_drop=0.5):
        super().__init__()
        self.device = torch.device("cuda" if gpu != -1 else "cpu")

        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.in_drop = nn.Dropout(gcn_in_drop)
        self.gcn_drop = nn.Dropout(gcn_out_drop)

        # gcn layer
        self.W_0 = nn.ModuleList()
        self.W_r = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = in_dim if layer == 0 else mem_dim
            self.W_0.append(nn.Linear(input_dim, self.mem_dim).to(self.device))
            self.W_r.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

    def forward(self, nodes, adj):
        """

        :param nodes:  batch_size * node_size * node_emb
        :param adj:  batch_size * node_size * node_size
        :return:
        """
        batch_size = nodes.size(0)
        gcn_inputs = self.in_drop(nodes)

        # masks = []
        denoms = []
        for batch in range(batch_size):
            # if adj[batch]._values().size(0) == 0:
            #     denom = torch.zeros(nodes.size(1)).to(self.device)
            # else:
            denom = torch.sparse.sum(adj[batch], dim=1).to_dense()  # dev 上adj[batch]==kong
                # t_g = denom + torch.sparse.sum(adj[batch], dim=0).to_dense()
            denom += 1
                # mask = t_g.eq(0)
            denoms.append(denom)
                # masks.append(mask)
        denoms = torch.stack(denoms).unsqueeze(2)

        outputs = gcn_inputs
        for l in range(self.layers):
            gAxW = []
            bxW = self.W_r[l](gcn_inputs)
            for batch in range(batch_size):
                # if adj[batch]._values().size(0) == 0:
                #     gAxW.append(torch.zeros(gcn_inputs.size(1), gcn_inputs.size(2)).to(self.device))
                # else:
                    xW = bxW[batch]
                    AxW = torch.sparse.mm(adj[batch], xW)
                    gAxW.append(AxW)
            gAxW = torch.stack(gAxW)  # <16, 3344, 100>
            # gAxWs = F.relu(gAxW / denoms)
            gAxWs = F.relu((gAxW + self.W_0[l](gcn_inputs)) / denoms)  # self loop
            gcn_inputs = self.gcn_drop(gAxWs) if l < self.layers - 1 else gAxWs
        gcn_outputs = outputs + gcn_inputs  # 加入残差链接
        return gcn_outputs

class DSGCN(nn.Module):

    def __init__(self, in_dim, mem_dim, num_layers, gpu, gcn_in_drop=0.5, gcn_out_drop=0.5):
        super().__init__()
        self.device = torch.device("cuda" if gpu != -1 else "cpu")

        self.layers = num_layers
        self.mem_dim = mem_dim
        self.head_dim = self.mem_dim // self.layers
        self.in_dim = in_dim
        self.in_drop = nn.Dropout(gcn_in_drop)
        self.gcn_drop = nn.Dropout(gcn_out_drop)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim).to(self.device)

        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

    def forward(self, nodes, adj):
        """

        :param nodes:  batch_size * node_size * node_emb
        :param adj:  batch_size * node_size * node_size
        :return:
        """
        batch_size = nodes.size(0)
        gcn_inputs = self.in_drop(nodes)
        denoms = []
        for batch in range(batch_size):
            denom = torch.sparse.sum(adj[batch], dim=1).to_dense()
            denom += 1
            denoms.append(denom)
        denoms = torch.stack(denoms).unsqueeze(2)

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            gAxW = []
            bxW = self.weight_list[l](outputs)
            for batch in range(batch_size):
                xW = bxW[batch]
                AxW = torch.sparse.mm(adj[batch], xW)
                gAxW.append(AxW)
            gAxW = torch.stack(gAxW)  # <16, 3344, 100>
            gAxWs = F.relu((gAxW + self.weight_list[l](gcn_inputs)) / denoms)  # self loop
            cache_list.append(gAxWs)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxWs))
        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs  # 加入残差链接
        out = self.linear_output(gcn_outputs)
        return out

class SpGAT(nn.Module):
    def __init__(self, in_dim, mem_dim, num_layers, gpu,  dropout, alpha, gcn_type, rel_hidden_dim=100, nheads=1):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.num_layers = num_layers
        self.nheads = nheads
        for layer in range(num_layers):
            input_dim = in_dim if layer == 0 else mem_dim
            for head in range(nheads):
                if gcn_type == 'GAT_RELREP':
                    self.attentions.append(
                        SpGraphAttentionLayer(input_dim, mem_dim, dropout=dropout, alpha=alpha, concat=False, rel_feature=rel_hidden_dim))
                else:
                    self.attentions.append(SpGraphAttentionLayer(input_dim, mem_dim, dropout=dropout, alpha=alpha, concat=False))
        self.gcn_type = gcn_type
        if gcn_type == 'GAT_RELREP':
            self.rel_embedding = nn.Embedding(1061, rel_hidden_dim)
            nn.init.xavier_uniform_(self.rel_embedding.weight.data, gain=1.414)

    def forward(self, nodes, adj, nodes_query, adj_edges=None):
        """
        :param nodes:  batch_size * node_size * node_emb
        :param nodes_query: # 实体的query向量
        :param adj:  batch_size * node_size * node_size
        :return:
        """
        batch_size = nodes.size(0)
        output_hiddens = []

        for batch in range(batch_size):
            x = nodes[batch]
            x = F.dropout(x, self.dropout, training=self.training)
            x_query = nodes_query[batch]
            x_query = F.dropout(x_query, self.dropout, training=self.training)
            if torch.isnan(x_query).any():
                print("x_query", x_query[0])
            if torch.isnan(x).any():
                print("x")

            for layer in range(self.num_layers):
                xs = []
                for head in range(self.nheads):
                    if self.gcn_type == 'GAT_RELREP':
                        x = self.attentions[layer * self.nheads + head](x, x_query, adj[batch], adj_edges[batch],
                                                                        self.rel_embedding)
                    else:
                        x = self.attentions[layer * self.nheads + head](x, x_query, adj[batch])
                    xs.append(x)
                    if torch.isnan(x).any():
                        print("head", head)
                        print("x", x[0])
                        exit(-1)
                x = torch.mean(torch.stack(xs), dim=0)  # 直接多个头取mean
                x = F.dropout(x, self.dropout, training=self.training) if layer < self.num_layers - 1 else x
            # x= nn.functional.normalize(x, dim=-1)
            output_hiddens.append(x)
        output_hiddens = torch.stack(output_hiddens)
        output_hiddens = output_hiddens + nodes  # 残差连接
        return output_hiddens


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, rel_feature=0):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features + rel_feature)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, input_query, adj, adj_edge=None, rel_embedding=None):
        """

        :param input: node_size * -1
        :param input_query:  # 实体的query向量
        :param adj: node_size * node_size
        :return:
        """
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.coalesce().indices()

        h = torch.mm(input, self.W)
        h_q = torch.mm(input_query, self.W)
        # h: N x out
        if torch.isnan(h).any():
            print("h", h)
            print("input", input)
            print(self.W.data)
            exit(-1)
        # assert not torch.isnan(h).any()
        if adj_edge is not None:
            adj_edge_rep = rel_embedding(adj_edge.coalesce().values())
            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h_q[edge[0, :], :], h[edge[1, :], :], adj_edge_rep), dim=1).t()
            assert not torch.isnan(edge_h).any()
        else:
            edge_h = torch.cat((h_q[edge[0, :], :], h[edge[1, :], :]), dim=1).t()

        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def split_n_pad(nodes, section, pad=0):
    nodes = torch.split(nodes, section.tolist())
    nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
    return nodes