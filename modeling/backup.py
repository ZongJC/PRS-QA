from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                 dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size // 2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size // 2)
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 2 + n_ntype * 2, hidden_size),
                                                torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_size, hidden_size))
        # fixed

        self.k = k
        self.gnn_layers = nn.ModuleList(
            [GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def gdc(self, A: sp.csr_matrix, alpha: float, k: int):
        N = A.shape[0]
        A_csc = A.tocsc()
        # Self-loops
        A_loop = sp.eye(N) + A_csc

        # Symmetric transition matrix
        D_loop_vec = A_loop.sum(0).A1
        D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
        D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
        T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

        # PPR-based diffusion
        S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

        S_csr = S.tocsr()

        num_nodes = S_csr.shape[0]

        row_idx = np.arange(num_nodes)
        S_csr[S_csr.argsort(0)[:num_nodes - k], row_idx] = 0

        # Sparsify using threshold epsilon
        S_tilde = S_csr.multiply(S_csr >= eps)

        # Column-normalized transition matrix on graph S_tilde
        D_tilde_vec = S_tilde.sum(0).A1
        T_S = S_tilde / D_tilde_vec

        return T_S

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for _ in range(self.k):
            _X, edge_attr = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        return _X

    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2]

        # Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(
                node_type.device)  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
            # 在这个地方算出节点分数的embedding
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score))  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = A  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous()  # [`total_n_nodes`, d_n  `ode] where `total_n_nodes` = b_size * n_node

        original_edge_index = edge_index.clone()
        original_edge_type = edge_type.clone()

        A_coo = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].cpu(), edge_index[1].cpu())),
                              shape=(_X.shape[0], _X.shape[0]))
        A_csr = A_coo.tocsr()
        t_s = self.gdc(A_csr, 0.4, 0.0025)
        # [0.05, 0.2]0.15,0.05 /0.25 0.05 /0.25 0.005/0.35 0.0025/ 0.3 0.001(太差)/ 0.6, 0.0025/0.4,0.002(太差)/0.15,0.0025
        t_s_coo = sp.coo_matrix(t_s)  # t_s是matrix
        edge_index = np.vstack((t_s_coo.row, t_s_coo.col))  # 提取edge_index

        edge_map = {tuple(original_edge_index[:, i]): original_edge_type[i] for i in
                    range(original_edge_index.shape[1])}
        edge_type = torch.tensor(
            [edge_map.get((edge_index[0, i], edge_index[1, i]), self.n_etype + 1) for i in range(edge_index.shape[1])])

        _node_type = node_type.view(-1).contiguous()  # [`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0),
                                                                                     -1).contiguous()  # [`total_n_nodes`, dim]
        # _node_feature_extra 这个变量里面包含了节点类型嵌入和节点相关性分数的嵌入
        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)
        # 那么通过mp_helper函数实现了消息传递和消息注意力权重之间的计算，其中_X中包含了上一层的节点信息
        X = _X.view(node_type.size(0), node_type.size(1), -1)  # [batch_size, n_node, dim]
        # 将返回的节点进行重塑

        output = self.activation(self.Vh(H) + self.Vx(X))  # 在这个地方实现节点更新后并加上上一层的节点信息
        output = self.dropout(output)  # 最终的聚合了周围节点的新的节点向量

        return output


class QAGNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim,
                                         dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
        # 多层感知机，用于最终的输出预测。
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None,
                cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)  #（10，1024）两个问题，一个问题五个选项，所以是10，后面是维度。
        all_hidden_states:
        concept_ids: (batch_size, n_node) #（10，200）
        adj: edge_index, edge_type  #tuple：2 tenor（2，8892） tensor（8892）
        adj_lengths: (batch_size,) #tensor（10） tensor([80, 45, 91, 98, 82, 65, 40, 63, 39, 96], device='cuda:1')
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        # 把句节点连接到图数据上
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1)  # (batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:] - 1, emb_data)  # (batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1))  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(
            1)).float()  # 0 means masked out #[batch_size, n_node]
        # 每个问题有5个子图，一个子图长度是200，每个子图长度不一样，比如第一个子图只有80个有效节点，那么80个节点之后的节点被Mask掉。
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  # [batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  # [batch_size, ]
        # 计算每个批次中有效节点分数的平均绝对值，用作归一化分母。
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        # 使用计算出的平均绝对值对节点分数进行归一化。
        node_scores = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(
            1)  # 1 means masked out

        mask = mask | (node_type_ids == 3)  # pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)  # 其实是一个多头注意力池化，传入的q，k，mask计算

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        # 这个地方相当于qagnn论文中图2最后图pooling+Z节点信息+问答上下文嵌入信息的组合并投入到多层感知机中
        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        logits = self.fc(concat)

        return logits, pool_attn


class LM_QAGNN(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}):
        # k,图神经网络层数，n_ntype=4节点类型的数量（可能是one-hot然后类别是4个，那么形如0010），n_etype是关系数量,
        # n_concept是799273，concept_dim是200，concept_in_dim1024，n_attention_head是2，fc_dim200, n_fc_layer0， p_emb, p_gnn, p_fc都是dropout0.2，
        # pretrained_concept_emb=tensor(799273,1024)
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                             n_concept, concept_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                             init_range=init_range)
        # self.encoder.sent_dim是roberta的向量维度

    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        inputs（tuple10，一个元组中是（2，5，100））
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        # bs=2，nc=5
        # Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]  # 获取input最后两个元素，分别是变得索引，边的类型
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [
            x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x, []) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        # lm_inputs,概念id，节点类型id、节点分数、邻接矩阵长度、边索引和边类型
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))  # edge_index: [2, total_E]   edge_type: [total_E, ]

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        # sent_vecs和all_hidden_states分别代表句向量第一个token，和最后一层的向量
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc,
                                                                                  -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type


class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        print('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path,
                                                                                          model_type, model_name,
                                                                                          max_seq_length)
        # 返回问题id，标签，还有那5个模型经过tokenization后的数据
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type,
                                                                                    model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)  # 5
        self.num_choice = num_choice
        print('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path,
                                                                                              max_node_num, num_choice,
                                                                                              args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,
                                                                                          num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in
                   [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in
                   [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path,
                                                                                           model_type, model_name,
                                                                                           max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path,
                                                                                                max_node_num,
                                                                                                num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in
                       [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in
                           [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size,
                                                   train_indexes, self.train_qids, self.train_labels,
                                                   tensors0=self.train_encoder_data, tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size,
                                                   torch.arange(len(self.train_qids)), self.train_qids,
                                                   self.train_labels, tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                                   tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1,
                                                       self.eval_batch_size, self.inhouse_test_indexes, self.train_qids,
                                                       self.train_labels, tensors0=self.train_encoder_data,
                                                       tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1,
                                                       self.eval_batch_size, torch.arange(len(self.test_qids)),
                                                       self.test_qids, self.test_labels,
                                                       tensors0=self.test_encoder_data, tensors1=self.test_decoder_data,
                                                       adj_data=self.test_adj_data)


###############################################################################
############################### GNN architecture ##############################
###############################################################################

from torch.autograd import Variable


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype;
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N,39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]
        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 2)  # [E, 39] self.netype+1 包含一个额外的类型
        # self_edge_vec = torch.zeros(x.size(0), self.n_etype +2).to(edge_vec.device) #创建一个全零向量，大小在自环位置设置为1
        # self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src#头实体的类型
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        # self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        # self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        # edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        # headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_vec = edge_vec.to('cuda:1')

        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]
        # 添加一句把edge_index放到gpu上
        edge_index = torch.tensor(edge_index, dtype=torch.long, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        # Add self loops to edge_index
        # loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        # loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        # edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)  # x(2000,400)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        # 消息传递过程是通过这个propagate函数开始调用重写message函数，
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out, edge_embeddings

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim][56360,200]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim][56360,400]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim][56360,400]
        # print("edge_index.size()", edge_index.size())
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim] [E,4,dim]
        # 生成键通过节点嵌入+节点类型嵌入+节点相关性分数嵌入+消息
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim] [E,4,50]
        # 消息
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        # 生成查询，里面是节点嵌入，节点类型嵌入，节点相关性分数嵌入

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E

        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        # 这个是消息乘周围节点的注意力分数
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]


class PromptEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.2):
        super(PromptEmbedding, self).__init__()
        self.sent_dim = output_dim
        self.node_dim = output_dim
        self.node_num = 200
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = output_dim*2
        # self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
        #                          nn.GELU(),
        #                          nn.Linear(input_dim//2, output_dim)
        #                          )
        self.fc = nn.Linear(self.hidden_dim, self.sent_dim)
        self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True, bidirectional=True)

        self.domain_projector = nn.Sequential(
            nn.Linear(output_dim,input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2,input_dim)
        )

        self.self_attn_layer = nn.MultiheadAttention(output_dim, num_heads=num_heads , dropout=dropout)

    def forward(self, text_embeds, node_embeds):
        node_embeds = node_embeds.view(-1,self.node_num,self.sent_dim)
        sent_vecs,_ = self.rnn(text_embeds) # [batch_size, seq_len, 2*input_dim]
        text_prime = self.fc(sent_vecs) # [batch_size, seq_len, input_dim]
        text_prime = torch.mean(text_prime, dim=1) # [batch_size, input_dim]
        H2, _ = self.self_attn_layer(node_embeds, node_embeds, node_embeds)
        text_prime_transposed = text_prime.unsqueeze(2)
        #cross-modality Attention
        attention_scores = torch.bmm(H2, text_prime_transposed) / (self.sent_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=1)
        text_prime_expanded = text_prime.unsqueeze(1) # [batch_size, 1, sent_dim]
        H3 = torch.bmm(attention_scores, text_prime_expanded).squeeze(1) # [batch_size, sent_dim]
        H4 = torch.mean(H3 , dim=1)
        Z = self.domain_projector(H4)
        return Z

class Text2graph(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.2):
        super(Text2graph, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = self.d_k ** 0.5
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.h * self.d_v, bias=True)
        self.W_O = nn.Linear(in_features=self.out_dim, out_features=self.d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_q]
    # Output
    # out  : [batch_size, len_q, h * d_v]

    def forward(self, Q, K, V, mask=None):
        Q_copy = Q.clone()
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])  # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])  # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])  # [batch_size, len_k, h, d_v]

        Q = Q.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_q, self.d_k])  # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_k])  # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_v])  # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k])
            alpha = F.softmax(A.masked_fill(_mask == 0, float('-inf')), dim=2)  #在Q的维度上进行mask，[batch_size*h,len_q,len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        return out

class Conv1dBiLSTM(nn.Module):
    def __init__(self, in_channels, cnn_kernel_num, cnn_window_size, word_dim, dropout=0.2):
        super(Conv1dBiLSTM, self).__init__()
        self.hidden_dim = in_channels//2
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=cnn_kernel_num, kernel_size=cnn_window_size,
                              padding=(cnn_window_size - 1) // 2)

        self.rnn = nn.GRU(input_size=in_channels, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True)

        self.norm1 = nn.BatchNorm1d(word_dim)
        self.dropout = nn.Dropout(dropout)

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x_copy = x.clone()
        x_local = F.leaky_relu((self.conv(x.permute(0, 2, 1))))#[batch_size, cnn_kernel_num, seq_len]
        x_local = x_local.permute(0, 2, 1) #[batch_size, seq_len, cnn_kernel_num]
        x_context, _ = self.rnn(x_copy) #[batch_size, seq_len, hidden_dim*2]
        combined = x_local + x_context #[batch_size, seq_len, word_dim]
        combined = self.dropout(combined)
        combined = self.norm1(combined.permute(0, 2, 1))#对合并后的特征用BN
        combined = combined.permute(0, 2, 1)
        return combined

class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_factor):
        super().__init__()
        self.input_dim = input_dim
        self.down_sample = nn.Linear(input_dim, input_dim // reduction_factor)
        self.activation = F.gelu
        self.up_sample = nn.Linear(input_dim // reduction_factor, input_dim)

    def forward(self,x):
        output = self.down_sample(x)
        output = self.activation(output)
        return self.up_sample(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, h: int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.2):
        super(FusionModule, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = self.d_k ** 0.5
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.h * self.d_v, bias=True)
        self.W_O = nn.Linear(in_features=self.out_dim, out_features=self.d_model, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.adapter1 = Adapter(input_dim=d_model, reduction_factor=4)
        self.norm1 = nn.LayerNorm(d_model)
        self.adapter2 = Adapter(input_dim=d_model, reduction_factor=4)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 4, dropout=dropout)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_q]
    # Output
    # out  : [batch_size, len_q, h * d_v]

    def forward(self, Q, K, V, mask=None):
        Q_copy = Q.clone()
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])  # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])  # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])  # [batch_size, len_k, h, d_v]

        Q = Q.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.h, self.len_q, self.d_k])  # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.h, self.len_k, self.d_k])  # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.h, self.len_k, self.d_v])  # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.h]).view([batch_size * self.h, 1, self.len_k])
            alpha = F.softmax(A.masked_fill(_mask == 0, float('-inf')),
                              dim=2)  # 在Q的维度上进行mask，[batch_size*h,len_q,len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout1(self.W_O(out))
        out = self.adapter1(out) + Q_copy
        out = self.norm1(out)

        out_copy = out.clone()
        out = self.ffn(out)
        out = self.dropout2(out) + out_copy
        out = self.norm2(out)
        return out

class Graph2text(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.2):
        super(Graph2text, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = self.d_k ** 0.5
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.h * self.d_v, bias=True)
        self.W_O = nn.Linear(in_features=self.out_dim, out_features=self.d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    # Input
    # Q    : [batch_size, len_q, d_model]
    # K    : [batch_size, len_k, d_model]
    # V    : [batch_size, len_k, d_model]
    # mask : [batch_size, len_q]
    # Output
    # out  : [batch_size, len_q, h * d_v]

    def forward(self, Q, K, V, mask=None):
        Q_copy = Q.clone()
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view([batch_size, self.len_q, self.h, self.d_k])  # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, self.len_k, self.h, self.d_k])  # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, self.len_k, self.h, self.d_v])  # [batch_size, len_k, h, d_v]

        Q = Q.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_q, self.d_k])  # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_k])  # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view([batch_size * self.h, self.len_k, self.d_v])  # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.unsqueeze(1).repeat(1, self.h, 1) # [batch_size, h, len_q]
            _mask = _mask.view(batch_size * self.h, self.len_q) # [batch_size *h,len_q]
            alpha = F.softmax(A.masked_fill(_mask.unsqueeze(2) == 0, float('-inf')), dim=1)  #在Q的维度上进行mask，[batch_size*h,len_q,len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        return out


#6/11
def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra,sent_vecs, last_hidden_states=None, lm_mask=None,gnn_mask=None):
     for _ in range(self.k):
         _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
         _X = self.activation(_X)
         _X = F.dropout(_X, self.dropout_rate, training = self.training)


         X = _X.view(-1, self.node_num , self.hidden_size) # [batch_size, n_node, dim]
         text2graph = self.text2graph(X, last_hidden_states, last_hidden_states, lm_mask) #左边是文本，右边是图，然后右边的图作为Q，文本是K，V
         last_hidden_states_hid = self.MaskedBiLSTM(last_hidden_states, lm_mask) #形状是[10,200,200]
         New_X = self.Fusenet(text2graph, last_hidden_states_hid, last_hidden_states_hid, lm_mask) #形状是[10,200,200]
         New_text = self.graph2text(last_hidden_states_hid, New_X, New_X, gnn_mask) #形状是[10,100,200]
         last_hidden_states = New_text
         _X = New_X.view(-1, self.hidden_size) # [batch_size * n_node, dim]
         promptembeds = self.promptembeds(last_hidden_states, _X, gnn_mask, lm_mask)
         return _X ,promptembeds

#6/12
def dot_attention(self, edge_embeddings, sent_vecs, lm_mask):
       #计算前E条边和句向量的相关性
        weights = torch.bmm(edge_embeddings[:edge_num].unsqueeze(1), expanded_sent_vecs.unsqueeze(2)).squeeze() / (self.emb_dim ** 0.5)
        weights_split = torch.split(weights, edge_lengehs)
        weights_normalized = [F.softmax(w, dim=0) for w in weights_split]
        weights_final = torch.cat(weights_normalized)
        updated_edge_embeddings = edge_embeddings.clone()
        updated_edge_embeddings[:edge_num] = updated_edge_embeddings[:edge_num] * weights_final.unsqueeze(-1)
        edge_embeddings = updated_edge_embeddings

    def caculate_attention(self, Q, R, edge_lengths, dim):
        """
            计算并返回每个图中边的更新后特征，基于给定的问题矩阵Q和边特征矩阵R。
            参数:
            - Q: torch.Tensor, 形状为[N, emb_dim]的问题矩阵
            - R: torch.Tensor, 形状为[N, emb_dim]的边特征矩阵
            - lengths: List[int], 包含每个图的边数
            - dim: int, 特征维度，默认为200
            返回:
            - new_edge_features: torch.Tensor, 更新后的边特征矩阵，形状为[N, emb_dim]
            """
        # 初始化新的边特征列表，用于存放更新后的边特征

        new_edge_features = []
        # 计算每张图的注意力分数并更新边特征
        start_idx = 0
        for length in edge_lengths:
            # 提取当前图的Q和R的部分
            Q_current = Q[start_idx:start_idx + length]
            R_current = R[start_idx:start_idx + length]
            # 计算注意力分数，这里使用了Q与R转置的乘积，然后应用softmax并除以sqrt(dim)以稳定计算
            #attention_scores = torch.matmul(Q_current, R_current.t()) / (dim ** 0.5)  # 稳定化处理[q,e] [e,r] [q,r]
            attention_scores = torch.matmul(torch.matmul(Q_current, self.W), R_current.t()) / (dim ** 0.5) #[1,e] [e,e][1,e] [e,r] [1,r]
            attentions = F.softmax(attention_scores, dim=1)  # 对每一行（边）进行归一化[1,r]
            # 应用注意力权重到边特征上
            updated_R_current = torch.matmul(attentions, R_current) # 扩展维度以进行逐元素乘法[1,r] [r,e] [q,e]
            # 收集更新后的边特征
            new_edge_features.append(updated_R_current)
            # 更新起始索引
            start_idx += length
        # 合并所有图的更新后边特征
        new_edge_features = torch.cat(new_edge_features, dim=0)
        return new_edge_features

def calculate_attention(self, Q, R, edge_lengths, dim):
    """
    计算并返回每个图中边的更新后特征，基于给定的问题矩阵Q和边特征矩阵R。
    参数:
    - Q: torch.Tensor, 形状为[10, emb_dim]的问题矩阵
    - R: torch.Tensor, 形状为[N, emb_dim]的边特征矩阵
    - edge_lengths: List[int], 包含每个图的边数
    - dim: int, 特征维度，默认为200
    返回:
    - new_edge_features: torch.Tensor, 更新后的边特征矩阵，形状为[N, emb_dim]
    """
    # 初始化新的边特征列表，用于存放更新后的边特征
    new_edge_features = []
    # 计算每张图的注意力分数并更新边特征
    start_idx = 0
    for i, length in enumerate(edge_lengths):
        # 提取当前图的问题向量Q和边特征矩阵R的部分
        Q_current = Q[i].unsqueeze(0)  # 添加unsqueeze以确保与W的乘法兼容
        R_current = R[start_idx:start_idx + length]

        # 计算注意力分数，使用双线性注意力
        attention_scores = torch.matmul(torch.matmul(Q_current, self.W), R_current.t()) / (dim ** 0.5)
        attentions = F.softmax(attention_scores, dim=-1)  # 对最后一维（边）进行归一化

        # 应用注意力权重到边特征上
        expanded_attentions = attentions.unsqueeze(-1)  # 扩展维度以匹配R_current的形状
        updated_R_current = R_current * expanded_attentions.squeeze(0)  # 逐元素乘法，同时移除unsqueeze增加的维度

        # 收集更新后的边特征
        new_edge_features.append(updated_R_current)

        # 更新起始索引
        start_idx += length

    # 合并所有图的更新后边特征
    new_edge_features = torch.cat(new_edge_features, dim=0)
    return new_edge_features

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, sent_vecs_r, edge_lengths, last_hidden_states=None, lm_mask=None,gnn_mask=None):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra, sent_vecs_r, edge_lengths)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)

            X = _X.view(-1, self.node_num , self.hidden_size) # [batch_size, n_node, dim]

            last_hidden_states_hid = self.SelfAttention(last_hidden_states, last_hidden_states, last_hidden_states, lm_mask) #自注意力层
            text2graph = self.text2graph(X, last_hidden_states_hid, last_hidden_states_hid, lm_mask) #左边是文本，右边是图，然后右边的图作为Q，文本是K，V
            New_X = self.Fusenet(text2graph, last_hidden_states_hid, last_hidden_states_hid, lm_mask) #形状是[10,200,200]
            New_text = self.graph2text(last_hidden_states_hid, New_X, New_X, gnn_mask) #形状是[10,100,200]
            last_hidden_states = New_text
            _X = New_X.view(-1, self.hidden_size) # [batch_size * n_node, dim]
        promptembeds = self.promptembeds(last_hidden_states, _X, gnn_mask, lm_mask)
        return _X ,promptembeds

    def caculate_attention(self, Q, R, edge_lengths, dim):
        """
            计算并返回每个图中边的更新后特征，基于给定的问题矩阵Q和边特征矩阵R。
            参数:
            - Q: torch.Tensor, 形状为[N, emb_dim]的问题矩阵
            - R: torch.Tensor, 形状为[N, emb_dim]的边特征矩阵
            - lengths: List[int], 包含每个图的边数
            - dim: int, 特征维度，默认为200
            返回:
            - new_edge_features: torch.Tensor, 更新后的边特征矩阵，形状为[N, emb_dim]
            """
        # 初始化新的边特征列表，用于存放更新后的边特征

        new_edge_features = []
        # 计算每张图的注意力分数并更新边特征
        start_idx = 0
        for length in edge_lengths:
            # 提取当前图的Q和R的部分
            Q_current = Q[start_idx:start_idx + length]
            R_current = R[start_idx:start_idx + length]
            # 计算注意力分数，这里使用了Q与R转置的乘积，然后应用softmax并除以sqrt(dim)以稳定计算
            #attention_scores = torch.matmul(Q_current, R_current.t()) / (dim ** 0.5)  # 稳定化处理[q,e] [e,r] [q,r]
            attention_scores = torch.matmul(torch.matmul(Q_current, self.W), R_current.t()) / (dim ** 0.5) #[1,e] [e,e][1,e] [e,r] [1,r]
            attentions = F.softmax(attention_scores, dim=1)  # 对每一行（边）进行归一化[1,r]
            # 应用注意力权重到边特征上
            updated_R_current = torch.matmul(attentions, R_current) # 扩展维度以进行逐元素乘法[1,r] [r,e] [q,e]
            # 收集更新后的边特征
            new_edge_features.append(updated_R_current)
            # 更新起始索引
            start_idx += length
        # 合并所有图的更新后边特征
        new_edge_features = torch.cat(new_edge_features, dim=0)
        return new_edge_features



