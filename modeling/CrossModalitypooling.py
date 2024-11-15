import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.bmm(q,k.transpose(1,2)) / torch.sqrt(torch.tensor(embed_dim).float())
        attn_weights = self.softmax(attn_weights)
        output = torch.bmm(attn_weights, v)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.ffn1 = nn.Linear(input_dim,hidden_dim)
        self.gelu = nn.GELU()
        self.ffn2 = nn.Linear(input_dim,output_dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.gelu(x)
        x = self.ffn2(x)
        return x

class CrossModalityPooling(nn.Module):
    def __init__(self,graph_embed_dim, text_embed_dim, sentence_length, num_sentences):
        super(CrossModalityPooling, self).__init__()
        self.graph_self_attn = SelfAttention(graph_embed_dim)
        self.ffn1 = FeedForwardNetwork(text_embed_dim, 2 * text_embed_dim, graph_embed_dim)
        self.ffn2 = FeedForwardNetwork(text_embed_dim, 2 * text_embed_dim, graph_embed_dim)
        self.graph_embed_dim = graph_embed_dim

    def forward(self, graph_embeddings, text_embeddings):
        # 图自注意力
        graph_embeddings = self.graph_self_attn(graph_embeddings)

        # 转换文本嵌入
        text_prime = self.ffn1(F.gelu(self.ffn2(text_embeddings)))

        # 确保跨模态注意力的维度匹配
        assert text_prime.shape[-1] == self.graph_embed_dim  #"T' 和 H2 的维度不匹配."

        # 跨模态注意力
        attn_scores = torch.matmul(graph_embeddings, text_prime.permute(0, 2, 1)) / self.graph_embed_dim ** 0.5
        attn_probs = F.softmax(attn_scores, dim=-1)
        h3 = torch.matmul(attn_probs, text_prime)

        # 图级别嵌入的平均汇聚
        graph_level_embedding = torch.mean(h3, dim=1)

        return graph_level_embedding

class Text2graph(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int):
        super(Text2graph, self).__init__()
        self.h = h
        self.d_model = d_model
        self.len_q = len_q
        self.len_k = len_k
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = self.h * self.d_v
        self.attention_scalar = math.sqrt(float(self.d_k))
        self.W_Q = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_K = nn.Linear(in_features=d_model, out_features=self.h * self.d_k, bias=True)
        self.W_V = nn.Linear(in_features=d_model, out_features=self.h * self.d_v, bias=True)
        self.W_O = nn.Linear(in_features=self.out_dim, out_features=self.d_model, bias=True)

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
        out = self.W_O(out) + Q_copy
        return out

class Conv1dBiLSTM(nn.Module):
    def __init__(self, in_channels,cnn_kernel_num,word_dim,dropout=0.2):
        super(Conv1dBiLSTM, self).__init__()
        self.hidden_dim = 150
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_kernel_num // 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_kernel_num // 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_kernel_num // 3, kernel_size=5, padding=2)

        self.rnn = nn.GRU(input_size=in_channels, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True)

        self.linear = nn.Linear(cnn_kernel_num + self.hidden_dim * 2, word_dim)
        self.norm1 = nn.BatchNorm1d(cnn_kernel_num + self.hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_copy = x.clone()
        x_local = F.relu(torch.cat([self.conv1(x.permute(0, 2, 1)),
                                    self.conv2(x.permute(0, 2, 1)),
                                    self.conv3(x.permute(0, 2, 1))],dim=1))#[batch_size, cnn_kernel_num, seq_len]
        x_local = x_local.permute(0, 2, 1) #[batch_size, seq_len, cnn_kernel_num]
        x_context,_ = self.rnn(x_copy) #[batch_size, seq_len, hidden_dim*2]
        combined = torch.cat((x_local, x_context),dim=2)
        combined = self.norm1(combined.permute(0, 2, 1))#对合并后的特征用BN
        combined = combined.permute(0, 2, 1)
        combined = self.dropout(combined)
        output = self.linear(combined)
        return output
