import torch

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,sent_dim,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +2 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
        # fixed

        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.sent_for_graphembedding = nn.Sequential(
            nn.Linear(sent_dim, sent_dim//2),
            nn.GELU(),
            nn.Linear(sent_dim//2, output_size)
        )

        self.self_attn = nn.MultiheadAttention(input_size, num_heads=4, dropout=0.1)

    def gdc(self, A: np.ndarray, alpha: float, beta: float, k: int, node_relevance_scores: np.ndarray, adj_lengths):
        #0.2 50 0.55
        node_relevance_scores = node_relevance_scores.squeeze(2)
        mean_norm = (torch.abs(node_relevance_scores)).sum(dim=1) / adj_lengths
        node_relevance_scores = node_relevance_scores / (mean_norm.unsqueeze(1) + 1e-05)
        node_relevance_scores = node_relevance_scores.unsqueeze(2)
        node_relevance_scores = node_relevance_scores.view(-1, node_relevance_scores.size(2)).contiguous()
        node_relevance_scores = node_relevance_scores.cpu().numpy()
        num_nodes = A.shape[0]

        # Step 1: 加入自环并构建初始的A_tilde
        A_tilde = A + np.eye(num_nodes)

        # Step 2: 度矩阵的逆平方根
        D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))

        # Step 3: 融合节点度信息和节点相关性信息
        # 将节点相关性分数转换为对角矩阵形式，与度矩阵结合
        relevance_diag = np.diag(node_relevance_scores)

        # 重新定义H，结合了节点度和节点相关性信息
        combined_info = beta * D_tilde + (1 - beta) * relevance_diag
        H = combined_info @ A_tilde @ combined_info

        # Step 4: 计算扩散矩阵,在求逆前向矩阵添加正则化技术。
        S = alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H + 1e-5 * np.eye(num_nodes))

        # Step 5: 截断处理保持前k个重要连接
        num_node = S.shape[0]
        row_idx = np.arange(num_node)
        S[S.argsort(axis=0)[:num_node - k], row_idx] = 0

        # Step 6: 归一化
        norm = S.sum(axis=0)
        norm[norm <= 0] = 1
        T = S / norm

        return T

    def get_adj_matrix(self, _X, edge_index):
        num_nodes = _X.shape[0]
        adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
        for i, j in zip(edge_index[0], edge_index[1]):
            adj_matrix[i, j] = 1.
        return adj_matrix

    def get_graphembedding(self, _X, sent_vec_graph):
        _X = _X.view(_X.size(0)//self.hidden_size, -1, _X.size(1))
        H2, _ = self.self_attn(_X,_X,_X)

        T_prime = torch.unsqueeze(sent_vec_graph, 1)
        H3=torch.softmax(torch.bmm(H2, T_prime.transpose(1,2))/math.sqrt(H2.size(-1)),dim=-1)

        H3=torch.bmm(H3, T_prime)
        H4=torch.max(H3,dim=1)

        return H4

    def mp_helper(self, _X, edge_index, edge_type, edges_attr, _node_type, _node_feature_extra, sent_vec_graph):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, edges_attr, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)

            #get graph embeeding
            #graph_emb = self.get_graphembedding(_X, sent_vec_graph)
            #history.append(graph_emb)
        return _X


    def forward(self, H, A, node_type, node_score, sent_vecs, node_scores_copy, adj_lengths, cache_output=False,
                batch=None, last_hidden_states=None, lm_mask=None):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
            #在这个地方算出节点分数的embedding
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_n  `ode] where `total_n_nodes` = b_size * n_node

        original_edge_index = edge_index.clone().transpose(0,1) # [total_E, 2]
        original_edge_type = edge_type.clone()

        A = self.get_adj_matrix(_X, edge_index)
        T = self.gdc(A, 0.3, 0.5, 100, node_scores_copy, adj_lengths) ##0.2 50 0.55 #0.4 100 0.5
        edges_i = []
        edges_j = []
        edges_attr = []
        for i, row in enumerate(T):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edges_attr.append(T[i,j])
        edge_index = [edges_i,edges_j]
        edge_index = torch.tensor(edge_index,dtype=torch.int64).to(node_type.device)
        edge_index_trans = edge_index.transpose(0,1)
        edges_attr = torch.tensor(edges_attr,dtype=torch.float16).to(node_type.device)
        #[0.05, 0.2]0.15,0.05 /0.25 0.05 /0.25 0.005/0.35 0.0025/ 0.3 0.001(太差)/ 0.6, 0.0025/0.4,0.002(太差)/0.15,0.0025

        edge_map = {}
        original_edge_index_tuples = [tuple(original_edge_index[i].tolist()) for i in range(original_edge_index.shape[0])]
        original_edge_type = original_edge_type.tolist()
        for key,value in zip(original_edge_index_tuples,original_edge_type):
            edge_map[key] = value

        edges_type = []
        for i in range(edge_index.shape[1]):
            edge_tuple = tuple(edge_index_trans[i].tolist())
            if edge_tuple in edge_map.keys():
                edges_type.append(edge_map.get(tuple(edge_index_trans[i].tolist())))
            else:
                if edge_tuple[0] == edge_tuple[1]:
                    edges_type.append(self.n_etype)
                else:
                    edges_type.append(self.n_etype + 1)

        edges_type = torch.tensor(edges_type,dtype=torch.int64)
        edge_type = edges_type.to(node_type.device)

        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]
        #_node_feature_extra 这个变量里面包含了节点类型嵌入和节点相关性分数的嵌入

        sent_vec_graph = self.sent_for_graphembedding(sent_vecs)

        _X = self.mp_helper(_X, edge_index, edge_type, edges_attr, _node_type, _node_feature_extra, sent_vec_graph)
        #那么通过mp_helper函数实现了消息传递和消息注意力权重之间的计算，其中_X中包含了上一层的节点信息
        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]
        #将返回的节点进行重塑

        output = self.activation(self.Vh(H) + self.Vx(X))  #在这个地方实现节点更新后并加上上一层的节点信息
        output = self.dropout(output) #最终的聚合了周围节点的新的节点向量

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
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, sent_dim=sent_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)
       #多层感知机，用于最终的输出预测。
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


    def forward(self, sent_vecs ,concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None,
                cache_output=False, batch=None, last_hidden_states=None, attention_mask=None):
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
        #复制节点分数node_scores用于计算扩散矩阵
        node_scores_copy = node_scores.clone()
        #把句节点连接到图数据上
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        #小于这个长度的节点序列被设置为True，否则为false
        #每个问题有5个子图，一个子图长度是200，每个子图长度不一样，比如第一个子图只有80个有效节点，那么80个节点之后的节点被Mask掉。
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        #计算每个批次中有效节点分数的平均绝对值，用作归一化分母。
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        #使用计算出的平均绝对值对节点分数进行归一化。
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        last_hidden_states = self.activation(self.svec2nvec(last_hidden_states.to(sent_vecs.device)))
        #形状是(10,100,200)
        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores, sent_vecs, node_scores_copy, adj_lengths, batch=batch, last_hidden_states=last_hidden_states, lm_mask=attention_mask )

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) #pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)#其实是一个多头注意力池化，传入的q，k，mask计算

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
        #k,图神经网络层数，n_ntype=4节点类型的数量（可能是one-hot然后类别是4个，那么形如0010），n_etype是关系数量,
        #n_concept是799273，concept_dim是200，concept_in_dim1024，n_attention_head是2，fc_dim200, n_fc_layer0， p_emb, p_gnn, p_fc都是dropout0.2，
        #pretrained_concept_emb=tensor(799273,1024)
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range)
        #self.encoder.sent_dim是roberta的向量维度

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
        #bs=2，nc=5
        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]#获取input最后两个元素，分别是变得索引，边的类型
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        #lm_inputs,概念id，节点类型id、节点分数、邻接矩阵长度、边索引和边类型
        edge_index, edge_type, g_batch= self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        #g_batch[2000],也就是[10*200],10个图，每个图200个节点。
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        input_ids, attention_mask, token_type_ids,output_mask =lm_inputs

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)

        last_hidden_states = all_hidden_states[-1]
        #sent_vecs和all_hidden_states分别代表句向量第一个token，和最后一层的向量
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output,
                                    batch=g_batch.to(node_type_ids.device),
                                    last_hidden_states=last_hidden_states,
                                    attention_mask=attention_mask
                                    )
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            #edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            #edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )


    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)

        item = [torch.full((n_nodes,), i, dtype=torch.long) for i in range(n_examples)]
        item = torch.cat(item, dim=0)

        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type, item



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
        print ('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        #返回问题id，标签，还有那5个模型经过tokenization后的数据
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1) #5
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)


        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

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
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
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
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)





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

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        #For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, edges_attr, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N,39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]
        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +2) #[E, 39] self.netype+1 包含一个额外的类型
        #self_edge_vec = torch.zeros(x.size(0), self.n_etype +2).to(edge_vec.device) #创建一个全零向量，大小在自环位置设置为1
        #self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] #[E,] #head=src#头实体的类型
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        #self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        #self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        #edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        #headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_vec=edge_vec.to('cuda:1')

        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]
        #fixed
        edges_attr = edges_attr.view(-1,1)
        edge_embeddings = edges_attr * edge_embeddings #[E+N, emb_dim]
        #添加一句把edge_index放到gpu上
        edge_index = torch.tensor(edge_index, dtype=torch.long, device='cuda:1' if torch.cuda.is_available() else 'cpu')
        #Add self loops to edge_index
        #loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        #loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        #edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)  #x(2000,400)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) #[N, emb_dim]
        #消息传递过程是通过这个propagate函数开始调用重写message函数，
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        #print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim][56360,200]
        #print ("x_j.size()", x_j.size()) #[E, emb_dim][56360,400]
        #print ("x_i.size()", x_i.size()) #[E, emb_dim][56360,400]
        #print("edge_index.size()", edge_index.size())
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim] [E,4,dim]
        #生成键通过节点嵌入+节点类型嵌入+节点相关性分数嵌入+消息
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim] [E,4,50]
        #消息
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        #生成查询，里面是节点嵌入，节点类型嵌入，节点相关性分数嵌入

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) #[E, heads]
        src_node_index = edge_index[0] #[E,]
        alpha = softmax(scores, src_node_index) #[E, heads] #group by src side node
        self._alpha = alpha

        #adjust by outgoing degree of src
        E = edge_index.size(1)            #n_edges
        N = int(src_node_index.max()) + 1 #n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] #[E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E

        alpha = alpha * src_node_edge_count.unsqueeze(1) #[E, heads]

        out = msg * alpha.view(-1, self.head_count, 1) #[E, heads, _dim]
        #这个是消息乘周围节点的注意力分数
        return out.view(-1, self.head_count * self.dim_per_head)  #[E, emb_dim]
