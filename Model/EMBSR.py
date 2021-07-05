from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PairSelfAttentionLayer(nn.Module):
    def __init__(self, x_dim, hidden_size, dropout):
        """
        Operation-aware SelfAttention
        """
        super(PairSelfAttentionLayer, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attention_q = nn.Linear(self.x_dim, self.hidden_size)
        self.LN = nn.LayerNorm(self.hidden_size)
        self.self_atten_w1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.self_atten_w2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, q, k, v, p, a, mask=None):
        """
        :param q: B, L, d
        :param k: B, L, d
        :param v: B, L, d
        :param p: B, L, d
        :param a: B, L, L, d
        :param mask:
        :return:
        """
        q_ = self.dropout(torch.relu(self.attention_q(q)))
        k_ = k
        v_ = v
        k_ = k_ + p
        batch_, length, dim = q_.size(0), q_.size(1), q_.size(2)
        q__ = q_.unsqueeze(2).repeat(1, 1, length, 1)
        qk = torch.matmul(q_, k_.transpose(1, 2))
        qa = (q__ * a).sum(dim=-1)
        scores = (qk + qa) / math.sqrt(self.hidden_size)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -1e12)
        alpha = torch.softmax(scores, dim=-1)
        v_ = v_ + p
        alphav = torch.matmul(alpha, v_)
        alpha__ = alpha.unsqueeze(2).repeat(1, 1, dim, 1)
        alphaa = (alpha__ * a.transpose(2, 3)).sum(dim=-1)
        att_v = alphav + alphaa
        att_v = self.dropout(self.self_atten_w2(torch.relu(self.self_atten_w1(att_v)))) + att_v
        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)
        return c, att_v


class HierarchicalGNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(HierarchicalGNN, self).__init__()
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden, macro_items, micro_actions, action_len):

        batch_size, n_edges, n_actions = micro_actions.size(0), micro_actions.size(1), micro_actions.size(2)
        action_len_ = action_len.view(batch_size * n_edges)
        packed = pack_padded_sequence(micro_actions.view(batch_size * n_edges, n_actions, self.embedding_size),
                                      action_len_.tolist(), batch_first=True, enforce_sorted=False)
        _, micro_actions_h = self.gru(packed)  # 1, B*n_edges, dim
        micro_actions_h = micro_actions_h.view(batch_size, n_edges, self.embedding_size)  # B, n_edges, dim
        # macro_hidden = macro_items + micro_actions_h
        macro_hidden = torch.cat([macro_items, micro_actions_h], 2)
        _in = self.linear_edge_in(macro_hidden)
        _out = self.linear_edge_out(macro_hidden)
        input_in = torch.matmul(A[:, :, :n_edges], _in) + self.b_iah
        input_out = torch.matmul(A[:, :, n_edges:], _out) + self.b_ioh

        inputs = torch.cat([input_in, input_out], 2)  #

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden, macro_items, micro_actions, micro_len):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, macro_items, micro_actions, micro_len)
        return hidden


class EMBSR(nn.Module):

    def __init__(self, n_items, n_actions, n_pos, n_action_pairs, x_dim, a_dim, pos_dim, hidden_dim,
                 dropout_rate=0, alpha=12, step=1):
        super(EMBSR, self).__init__()
        self.item_embeddings = nn.Embedding(n_items, x_dim, padding_idx=0)
        self.action_embeddings = nn.Embedding(n_actions, a_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(n_pos, pos_dim, padding_idx=0)
        self.action_pairs_embeddings = nn.Embedding(n_action_pairs, a_dim, padding_idx=0)
        self.x_dim = x_dim

        self.dropout = nn.Dropout(dropout_rate)
        # self.input_size = x_dim + pos_dim + a_dim
        self.hidden_size = hidden_dim
        self.embedding_size = hidden_dim
        self.gnn = HierarchicalGNN(hidden_dim)
        self.pair_self_attention = PairSelfAttentionLayer(self.x_dim, self.hidden_size, self.dropout)

        # self.gru_relation = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.W_q_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_q_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_g = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_zero = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
        # parameters initialization
        self.step = step  #
        self._reset_parameters()
        self.alpha = alpha

    def _reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(0, 0.1)

    def forward(self, items, A, alias_inputs, item_seq_len, macro_items, micro_actions, micro_len, actions,
                action_pairs, pos):

        seq_len = alias_inputs.size(1) + 1
        actions = actions[:, :seq_len]
        pos = pos[:, :seq_len]
        action_pairs = action_pairs[:, :seq_len, :seq_len]
        hidden, action_pairs_embedding = self.item_embeddings(items), self.action_pairs_embeddings(
            action_pairs)
        actions_embedding = self.action_embeddings(actions)
        pos_embedding = self.pos_embeddings(pos)
        macro_items_embedding = self.item_embeddings(macro_items)  # B, n_edges, dim
        micro_actions_embedding = self.action_embeddings(micro_actions)  # B, n_edges, n_micro_a, dim
        mask = alias_inputs.gt(-1)
        h_0 = hidden
        hidden_mask = items.gt(0)
        length = torch.sum(hidden_mask, 1).unsqueeze(1)
        star_node = torch.sum(hidden, 1) / length  # B,d
        # star_0 = star_node
        for i in range(self.step):
            hidden, star_node = self.update_item_layer(hidden, star_node.squeeze(), A, macro_items_embedding,
                                                       micro_actions_embedding, micro_len)  # batch, hidden_size
        h_f = self.highway_network(hidden, h_0)

        # action_output, action_hidden = self.gru_relation(actions_embedding, None)
        alias_inputs = alias_inputs.masked_fill(mask == False, 0)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)
        new_mask = actions.gt(0)
        seq_hidden = torch.gather(h_f, dim=1, index=alias_inputs)
        seq_hidden = torch.cat([star_node, seq_hidden], dim=1)
        # self_input = seq_hidden + pos_embedding + actions_embedding
        self_input = seq_hidden + actions_embedding
        # item_seq_len 是没有special 的序列长度
        ht = self.gather_indexes(self_input, item_seq_len)

        _, outs = self.pair_self_attention(self_input, self_input, self_input, pos_embedding,
                                           action_pairs_embedding, new_mask)
        h_n = outs[:, 0, :]
        seq_out = self.linear_transform(torch.cat([h_n, ht], dim=1))
        sigma = torch.sigmoid(seq_out)
        seq_out = sigma * h_n + (1 - sigma) * ht
        c = seq_out.squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        l_emb = self.item_embeddings.weight[1:] / torch.norm(self.item_embeddings.weight[1:], dim=-1).unsqueeze(1)
        y = self.alpha * torch.matmul(l_c, l_emb.t())
        return y

    def update_item_layer(self, h_last, star_node_last, A, macro_items_e, micro_actions_e, micro_len):
        hidden = self.gnn(A, h_last, macro_items_e, micro_actions_e, micro_len)
        q_one = self.W_q_one(hidden)  # B, L, d
        k_one = self.W_k_one(star_node_last)  # B, d
        alpha_i = torch.bmm(q_one, k_one.unsqueeze(2)) / math.sqrt(self.embedding_size)  # B, L, 1
        new_h = (1 - alpha_i) * hidden + alpha_i * star_node_last.unsqueeze(1)  # B, L, d

        q_two = self.W_q_two(star_node_last)
        k_two = self.W_k_two(new_h)
        beta = torch.softmax(torch.bmm(k_two, q_two.unsqueeze(2)) / math.sqrt(self.embedding_size), 1)  # B, L, 1
        new_star_node = torch.bmm(beta.transpose(1, 2), new_h)  # B, 1, d

        return new_h, new_star_node

    def highway_network(self, hidden, h_0):
        g = torch.sigmoid(self.W_g(torch.cat((h_0, hidden), 2)))  # B,L,d
        h_f = g * h_0 + (1 - g) * hidden
        return h_f

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
