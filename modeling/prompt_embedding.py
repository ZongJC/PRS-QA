import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.2):
        super(PromptEmbedding, self).__init__()
        self.sent_dim = output_dim
        self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                 nn.GELU(),
                                 nn.Linear(input_dim//2, output_dim)
                                 )

        self.domain_projector = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.GELU(),
            nn.Linear(output_dim,output_dim)
        )

        self.self_attn_layer = nn.MultiheadAttention(output_dim, num_heads=num_heads , dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent_vecs, node_embeds):
        text_prime = self.fc1(sent_vecs)
        H2, _ = self.self_attn_layer(node_embeds, node_embeds, node_embeds)
        H2 = H2.view(-1, self.node_num, self.node_dim)
        text_prime_transposed = text_prime.unsqueeze(2)
        #cross-modality Attention
        attention_scores = torch.bmm(H2, text_prime_transposed) / (self.sent_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=1)

        text_prime_expanded = text_prime.unsqueeze(1) # [batch_size, 1, sent_dim]

        H3 = torch.bmm(attention_scores, text_prime_expanded).squeeze(1) # [batch_size, sent_dim]

        H4 = torch.mean(H3 , dim=1)

        Z = self.domain_projector(H4)

        return Z

    class PromptEmbedding(nn.Module):
        def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
            super(PromptEmbedding, self).__init__()
            self.sent_dim = output_dim
            self.node_dim = output_dim
            self.node_num = 200
            self.dropout = nn.Dropout(dropout)
            self.hidden_dim = output_dim * 2
            self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim // 2),
                                     nn.GELU(),
                                     nn.Linear(input_dim // 2, output_dim)
                                     )
            self.fc = nn.Linear(self.hidden_dim, self.sent_dim)
            # self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True, bidirectional=True)

            self.domain_projector = nn.Sequential(
                nn.Linear(output_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, input_dim)
            )

            self.self_attn_layer = nn.MultiheadAttention(output_dim, num_heads=num_heads, dropout=dropout)

        def forward(self, text_embeds, node_embeds, gnn_mask, lm_mask):
            node_embeds = node_embeds.view(-1, self.node_num, self.sent_dim)
            # text_prime = torch.mean(text_embeds, dim=1) # [batch_size, input_dim]
            H2, _ = self.self_attn_layer(node_embeds, node_embeds, node_embeds, attn_mask=gnn_mask)
            text_prime_transposed = text_prime.unsqueeze(2)
            # cross-modality Attention
            attention_scores = torch.bmm(H2, text_prime_transposed) / (self.sent_dim ** 0.5)
            attention_scores = F.softmax(attention_scores, dim=1)
            text_prime_expanded = text_prime.unsqueeze(1)  # [batch_size, 1, sent_dim]
            H3 = torch.bmm(attention_scores, text_prime_expanded).squeeze(1)  # [batch_size, sent_dim]
            H4 = torch.mean(H3, dim=1)
            Z = self.domain_projector(H4)
            return Z