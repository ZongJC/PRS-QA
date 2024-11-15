import torch

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from torch import nn
import numpy as np
from utils.sag_pool import SAGPooling
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,sent_dim,
                    dropout=0.1,pooling_ratio=0.96):
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

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
        # self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
        #                                         torch.nn.LayerNorm(hidden_size), torch.nn.ReLU(),
        #                                         torch.nn.Linear(hidden_size, hidden_size))
        #生成边的特征表示

        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.pooling_ratio = pooling_ratio
        self.SAGPools = nn.ModuleList([SAGPooling(hidden_size, ratio=self.pooling_ratio, GNN=GATConv, nonlinearity=torch.tanh) for _ in range(k)])

        self.promptembeds = PromptEmbedding(input_dim=sent_dim ,output_dim=input_size)
        # self.text2graph = Text2graph(h=4,d_model=hidden_size,len_q=200,len_k=100,d_k=hidden_size,d_v=hidden_size)
        # self.Fusenet = FusionModule(h=4,d_model=hidden_size,len_q=200,len_k=100,d_k=hidden_size,d_v=hidden_size)
        # self.graph2text = Graph2text(h=8,d_model=hidden_size,len_q=100,len_k=200,d_k=hidden_size,d_v=hidden_size)
        # self.SelfAttention = MultiHeadselfAttention(h=4,d_model=hidden_size,len_q=100,len_k=100,d_k=hidden_size, d_v=hidden_size)

        self.text2graph = Text2graph(h=4, d_model=hidden_size, len_q=200, len_k=512, d_k=hidden_size, d_v=hidden_size)
        self.Fusenet = FusionModule(h=4, d_model=hidden_size, len_q=200, len_k=512, d_k=hidden_size, d_v=hidden_size)
        self.graph2text = Graph2text(h=8, d_model=hidden_size, len_q=512, len_k=200, d_k=hidden_size, d_v=hidden_size)
        self.SelfAttention = MultiHeadselfAttention(h=4, d_model=hidden_size, len_q=512, len_k=512, d_k=hidden_size,d_v=hidden_size)

    # def filter_Pool(self, x, mask):
    #     mask_expanded = mask.unsqueeze(2).type_as(x)
    #     mask_features = x * mask_expanded
    #     sum_features = torch.sum(mask_features, dim=1) # [batch_size, node_dim]
    #     #计算每个样本的有效节点数
    #     valid_node_num = torch.sum(mask, dim=1, keepdim=True).type_as(x)
    #     eps = torch.finfo(torch.float16).eps  # 获取float16的最小正值
    #     valid_node_counts = torch.clamp(valid_node_num, min=eps)
    #     average_data = sum_features / valid_node_counts
    #     return average_data
    def count_edges_per_subgraph(self, edge_lengths):
        counts = torch.bincount(edge_lengths)
        return counts

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, sent_vecs_r, edge_lengths, batch, last_hidden_states=None, lm_mask=None,gnn_mask=None):

        _batch_size, _n_nodes = gnn_mask.size()

        for _ in range(self.k):
            #edge_piece = self.filter_Pool(last_hidden_states,lm_mask)
            #在这里计算每个子图的边的数量
            edge_counts = self.count_edges_per_subgraph(edge_lengths)
            edge_counts = edge_counts.clone().detach()

            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra, sent_vecs_r, edge_counts)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)

            X = _X.view(_batch_size, -1 , self.hidden_size) # [batch_size, n_node, dim]

            node_num = X.size(1)

            last_hidden_states_hid = self.SelfAttention(last_hidden_states, last_hidden_states, last_hidden_states, lm_mask) #自注意力层
            text2graph = self.text2graph(X, last_hidden_states_hid, last_hidden_states_hid, node_num, lm_mask) #左边是文本，右边是图，然后右边的图作为Q，文本是K，V
            New_X = self.Fusenet(text2graph, last_hidden_states_hid, last_hidden_states_hid, node_num, lm_mask) #形状是[10,200,200]
            New_text, lm_node_scores = self.graph2text(last_hidden_states_hid, New_X, New_X, node_num, gnn_mask, return_attn=True) #形状是[10,100,200]
            last_hidden_states = New_text

            _X = New_X.view(-1, self.hidden_size) # [batch_size * n_node, dim]

            #动态剪枝
            lm_node_scores = torch.sum(lm_node_scores, -1).view(-1)
            gnn_mask = gnn_mask.view(-1)

            _X, edge_index, edge_type, _node_type, batch, perm, score, edge_lengths = self.SAGPools[_](_X, lm_node_scores, edge_index, edge_type, _node_type, batch, edge_lengths)
            #获取了新的edge_index,_X少了从2000变为1880正好是2000*0.94
            gnn_mask = gnn_mask[perm]
            gnn_mask = gnn_mask.view(_batch_size, -1)

        gnn_mask = gnn_mask.view(_batch_size,-1)
        promptembeds = self.promptembeds(last_hidden_states, _X, gnn_mask, lm_mask)
        return _X, batch, gnn_mask, promptembeds


    def forward(self, H, A, edge_lengths, batch, node_type, node_score, sent_vecs_r, cache_output=False, last_hidden_states=None, lm_mask=None, gnn_mask=None):
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
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        _X, batch, new_gnn_mask, promptembeds = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra, sent_vecs_r, edge_lengths,batch,
                                          last_hidden_states=last_hidden_states, lm_mask=lm_mask, gnn_mask=gnn_mask)

        num_nodes = scatter_add(batch.new_ones(_X.size(0)), batch, dim=0)

        X = _X.view(node_type.size(0), num_nodes[0], -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vx(X)) # 与qagnn不同去掉上一层的节点信息
        output = self.dropout(output)

        return output, new_gnn_mask, promptembeds


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


    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, edge_lengths, batch, emb_data=None, cache_output=False,last_hidden_states=None, attention_mask=None):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)#取出所有行，并且从每一行的第二个元素开始取到最后一个元素
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1] [前3个里面，第一个200中有49个第一个是0，第二个里面有45个，第三个43个]
        #_masktensor([ 50,  46,  43,  44,  46, 111,  71, 112,  57, 122], device='cuda:1')比这几个大的是0

        gnn_mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) <= adj_lengths.unsqueeze(1)

        last_hidden_states = self.activation(self.svec2nvec(last_hidden_states))#把最后一层的输出几点的向量维度变成dim_node

        sent_vecs_r = self.activation(self.svec2nvec(sent_vecs)) #把句子向量维度变成200

        gnn_output, new_gnn_mask, promptembeds = self.gnn(gnn_input, adj, edge_lengths, batch, node_type_ids, node_scores, sent_vecs_r,last_hidden_states=last_hidden_states, lm_mask=attention_mask,gnn_mask=gnn_mask)

        pool_mask = ~new_gnn_mask

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        # mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out
        #
        # mask = mask | (node_type_ids == 3) #pool over all KG nodes
        # mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs = sent_vecs + promptembeds

        #sent_vecs = self.activation(self.transformprompt(torch.cat((sent_vecs, promptembeds), dim=1)))

        #sent_vecs = self.transformprompt((torch.cat((sent_vecs, promptembeds), dim=1)))

        #sent_vecs_for_pooler = sent_vecs
        sent_vecs_for_pooler = sent_vecs

        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, pool_mask)


        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn


class LM_QAGNN(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range)


    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
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

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        edge_index, edge_type, edge_lengths, item = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        input_ids, attention_mask, token_type_ids, output_mask = lm_inputs

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        last_hidden_states = all_hidden_states[-1]

        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj, edge_lengths,batch=item.to(node_type_ids.device),
                                    emb_data=None, cache_output=cache_output,
                                    last_hidden_states=last_hidden_states.to(node_type_ids.device),
                                    attention_mask=attention_mask.to(node_type_ids.device))
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

        item = [torch.full((n_nodes, ), i, dtype = torch.long) for i in range(n_examples)]
        item = torch.cat(item, dim=0)

        edge_item = [torch.full((len(edge_types),), i, dtype=torch.long) for i, edge_types in enumerate(edge_type_init)]
        edge_lengths = torch.cat(edge_item,dim = 0)

        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        #edge_lengths = [i.size(0) for i in edge_type_init]
        return edge_index, edge_type, edge_lengths, item
     #在这个地方item用于分开哪些点是batch （2000），每200个是一组，然后tensor是（0，0，0……9，9，9）



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
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
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
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
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
        self.edge_encoder_again = torch.nn.Sequential(torch.nn.Linear(emb_dim * 2, emb_dim),
                                                      torch.nn.BatchNorm1d(emb_dim),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(emb_dim, emb_dim))

        # self.edge_encoder_again = torch.nn.Sequential(torch.nn.Linear(emb_dim * 2, emb_dim),
        #                                               torch.nn.LayerNorm(emb_dim),
        #                                               torch.nn.ReLU(),
        #                                               torch.nn.Linear(emb_dim, emb_dim))

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        # self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        # self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        # self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)
        self.linear_key = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(1 * emb_dim, head_count * self.dim_per_head)

        self.W = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.W.data)

        self._alpha = None
        #For final MLP
        #self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.LayerNorm(emb_dim),torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

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

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, sent_vec_r, edge_lengehs, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim] 节点相关性分数
        # sent_vec [bs, emb_dim] 句子向量
        expanded_sent_vecs = []
        #处理句向量
        for i,length in enumerate(edge_lengehs):
            repeated_sent_vec = sent_vec_r[i].repeat(length,1)#在第0个维度上重复length次，在第1个维度上不进行重复
            expanded_sent_vecs.append(repeated_sent_vec)

        #将所有扩展后的向量连接起来形成一个新的向量
        expanded_sent_vecs = torch.cat(expanded_sent_vecs,dim=0)

        #节点特征
        edge_num = edge_index.size(1)
        x_head, x_tail = x[edge_index[0]], x[edge_index[1]] #[E, emb_dim]
        x_fact = x_tail - x_head #边特征
        #将edge_type从gpu移到cpu，然后转换为numpy数组
        edge_type_np = edge_type.cpu().numpy()
        type_counts = np.bincount(edge_type_np)  # 统计每个边类型出现的次数
        x_fact_np = x_fact.detach().cpu().numpy()
        result = np.zeros_like(x_fact_np)
        for edge_type_value in np.unique(edge_type_np):
            #找到当前类型对应的索引
            indices = np.where(edge_type_np == edge_type_value)[0]
            #对应特征求和
            sum_features = np.sum(x_fact_np[indices],axis=0)
            #计算平均并赋值
            result[indices] = sum_features / type_counts[edge_type_value]
        #将result转回为tesor并放回gpu
        result_tensor = torch.tensor(result,dtype=x_fact.dtype,device=x_fact.device) #(9176,200) ,[E,emb_dim]

        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +1) #[E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        edge_embeddings_one = edge_embeddings[:edge_num]
        edge_embeddings_two = edge_embeddings[edge_num:]
        concatenated = torch.cat((edge_embeddings_one, result_tensor),dim=1)
        new_values = self.edge_encoder_again(concatenated) #[E,emb_dim]
        new_edge_embeddings = torch.cat([new_values, edge_embeddings_two], dim=0)
        edge_embeddings = new_edge_embeddings
        #计算前E条边和句向量的相关性
        Relation = edge_embeddings[:edge_num] #[E, emb_dim]
        Relation_others = edge_embeddings[edge_num:]
        updated_Relation = self.caculate_attention(Q=expanded_sent_vecs, R=Relation, edge_lengths=edge_lengehs, dim=self.emb_dim)
        edge_embeddings = torch.cat([updated_Relation, Relation_others],dim=0)

        # remove self loops
        edge_index, _ = remove_self_loops(edge_index)

        #Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        #x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 1 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

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
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]