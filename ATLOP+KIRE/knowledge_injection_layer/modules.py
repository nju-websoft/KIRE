from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Combineloss(nn.Module):
    def __init__(self, config):
        super(Combineloss, self).__init__()
        self.config = config
        self.small_inf = 1e-30

    def forward(self, re_loss, coref_loss, kg_loss):
        loss = self.config.alpha_re * re_loss
        if coref_loss is not None:
            loss += self.config.alpha_coref * coref_loss

        if kg_loss is not None:
            loss += self.config.alpha_kg * kg_loss
        return loss


class SpGAT(nn.Module):
    def __init__(self, in_dim, mem_dim, num_layers, dropout, alpha, rel_hidden_dim=100, nheads=1):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList()
        self.num_layers = num_layers
        self.nheads = nheads
        for layer in range(num_layers):
            input_dim = in_dim if layer == 0 else mem_dim
            for head in range(nheads):
                self.attentions.append(
                    SpGraphAttentionLayer(input_dim, mem_dim, dropout=dropout, alpha=alpha, concat=False, rel_feature=rel_hidden_dim))

        self.rel_embedding = nn.Embedding(1061, rel_hidden_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data, gain=1.414)

    def forward(self, nodes, adj, nodes_query, adj_edges=None):
        """
        :param nodes:  batch_size * node_size * node_emb
        :param nodes_query:
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
            for layer in range(self.num_layers):
                xs = []
                for head in range(self.nheads):
                    x = self.attentions[layer*self.nheads + head](x, x_query, adj[batch], adj_edges[batch], self.rel_embedding)
                    xs.append(x)
                x = torch.mean(torch.stack(xs), dim=0)
                x = F.dropout(x, self.dropout, training=self.training) if layer < self.num_layers - 1 else x
            output_hiddens.append(x)
        output_hiddens = torch.stack(output_hiddens)
        output_hiddens = output_hiddens + nodes
        return output_hiddens


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        # a = a.to_dense()
        # b = b.to_dense()
        c = torch.matmul(a, b)
        # c = c.half()
        return c
        # return torch.matmul(a, b)

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
        :param input_query:
        :param adj: node_size * node_size
        :return:
        """
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = adj.coalesce().indices()

        h = torch.mm(input, self.W)
        h_q = torch.mm(input_query, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        if adj_edge is not None:
            adj_edge_rep = rel_embedding(adj_edge.coalesce().values())
            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h_q[edge[0, :], :], h[edge[1, :], :], adj_edge_rep), dim=1).t()
            assert not torch.isnan(edge_h).any()
        else:
            edge_h = torch.cat((h_q[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        # print(edge_h)
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        # print(edge_e)
        # if torch.isnan(edge_e).any():
        #     print(edge_h)
        #     print(edge_e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        # print(e_rowsum)
        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime)
        # print(h_prime)
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