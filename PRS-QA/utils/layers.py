import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils.utils import freeze_net


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class TypedLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_type):
        super().__init__(in_features, n_type * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_type = n_type

    def forward(self, X, type_ids=None):
        """
        X: tensor of shape (*, in_features)
        type_ids: long tensor of shape (*)
        """
        output = super().forward(X)
        if type_ids is None:
            return output
        output_shape = output.size()[:-1] + (self.out_features,)
        output = output.view(-1, self.n_type, self.out_features)
        idx = torch.arange(output.size(0), dtype=torch.long, device=type_ids.device)
        output = output[idx, type_ids.view(-1)].view(*output_shape)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_p = emb_p
        self.input_p = input_p
        self.hidden_p = hidden_p
        self.output_p = output_p
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = np.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
                           num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                           batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bz, full_length = inputs.size()
        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(lstm_inputs)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
        rnn_outputs = self.output_dropout(rnn_outputs)
        return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs


class TripleEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, input_p, output_p, hidden_p, num_layers, bidirectional=True, pad=False,
                 concept_emb=None, relation_emb=None
                 ):
        super().__init__()
        if pad:
            raise NotImplementedError
        self.input_p = input_p
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.cpt_emb = concept_emb
        self.rel_emb = relation_emb
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=(hidden_dim // 2 if self.bidirectional else hidden_dim),
                          num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)

        returns: (batch_size, h_dim(*2))
        '''
        bz, sl = inputs.size()
        h, r, t = torch.chunk(inputs, 3, dim=1)  # (bz, 1)

        h, t = self.input_dropout(self.cpt_emb(h)), self.input_dropout(self.cpt_emb(t))  # (bz, 1, dim)
        r = self.input_dropout(self.rel_emb(r))
        inputs = torch.cat((h, r, t), dim=1)  # (bz, 3, dim)
        rnn_outputs, _ = self.rnn(inputs)  # (bz, 3, dim)
        if self.bidirectional:
            outputs_f, outputs_b = torch.chunk(rnn_outputs, 2, dim=2)
            outputs = torch.cat((outputs_f[:, -1, :], outputs_b[:, 0, :]), 1)  # (bz, 2 * h_dim)
        else:
            outputs = rnn_outputs[:, -1, :]

        return self.output_dropout(outputs)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class TypedMultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1, n_type=1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = TypedLinear(d_k_original, n_head * self.d_k, n_type)
        self.w_vs = TypedLinear(d_k_original, n_head * self.d_v, n_type)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, type_ids=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: bool tensor of shape (b, l) (optional, default None)
        type_ids: long tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k, type_ids=type_ids).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k, type_ids=type_ids).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class BilinearAttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))
        attn = self.softmax(attn.squeeze(-1))
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)
        return pooled, attn


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # # To limit numerical errors from large vector elements outside the mask, we zero these out.
            # result = nn.functional.softmax(vector * mask, dim=dim)
            # result = result * mask
            # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            raise NotImplementedError
        else:
            masked_vector = vector.masked_fill(mask.to(dtype=torch.uint8), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
            result = result * (1 - mask)
    return result


class DiffTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        """
        x: tensor of shape (batch_size, n_node)
        k: int
        returns: tensor of shape (batch_size, n_node)
        """
        bs, _ = x.size()
        _, topk_indexes = x.topk(k, 1)  # (batch_size, k)
        output = x.new_zeros(x.size())
        ri = torch.arange(bs).unsqueeze(1).expand(bs, k).contiguous().view(-1)
        output[ri, topk_indexes.view(-1)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)

class MultiHeadselfAttention(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.1):
        super(MultiHeadselfAttention, self).__init__()
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
        self.norm = nn.LayerNorm(d_model)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)
    # def initialize(self):
    #     nn.init.kaiming_uniform_(self.W_Q.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_Q.bias)
    #     nn.init.kaiming_uniform_(self.W_K.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_K.bias)
    #     nn.init.kaiming_uniform_(self.W_V.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_V.bias)
    #     nn.init.kaiming_uniform_(self.W_O.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_O.bias)

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
            alpha = F.softmax(A, dim=2) # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        out = self.norm(out)
        return out

class selfAttention_prompt(nn.Module):
    def __init__(self, d_model: int, num_head:int, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_dim = self.d_model * self.num_head
        self.attention_scalar = self.d_model ** 0.5
        self.W_Q = nn.Linear(in_features=self.d_model, out_features=self.num_head * self.d_model, bias=True)
        self.W_K = nn.Linear(in_features=self.d_model, out_features=self.num_head * self.d_model, bias=True)
        self.W_V = nn.Linear(in_features=self.d_model, out_features=self.num_head * self.d_model, bias=True)
        self.W_O = nn.Linear(in_features=self.out_dim, out_features=self.d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def initialize(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.zeros_(self.W_K.bias)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_V.bias)
        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    def forward(self, Q, mask=None):
        Q_copy = Q.clone()
        batch_size = Q.size(0)
        len_q = Q.size(1)
        len_k = Q.size(1)
        K = Q
        V = Q
        Q = self.W_Q(Q).view([batch_size, len_q, self.num_head, self.d_model])  # [batch_size, len_q, h, d_k]
        K = self.W_K(K).view([batch_size, len_k, self.num_head, self.d_model])  # [batch_size, len_k, h, d_k]
        V = self.W_V(V).view([batch_size, len_k, self.num_head, self.d_model])  # [batch_size, len_k, h, d_v]

        Q = Q.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.num_head, len_q, self.d_model])  # [batch_size * h, len_q, d_k]
        K = K.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.num_head, len_k, self.d_model])  # [batch_size * h, len_k, d_k]
        V = V.permute(0, 2, 1, 3).contiguous().view(
            [batch_size * self.num_head, len_k, self.d_model])  # [batch_size * h, len_k, d_v]
        A = torch.bmm(Q, K.permute(0, 2, 1).contiguous()) / self.attention_scalar  # [batch_size * h, len_q, len_k]
        if mask != None:
            _mask = mask.repeat([1, self.num_head]).view([batch_size * self.num_head, 1, len_k])
            alpha = F.softmax(A.masked_fill(_mask == 0, float('-inf')),dim=2)  # 在Q的维度上进行mask，[batch_size*h,len_q,len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.num_head, len_q, self.d_model])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        out = self.norm(out)
        return out


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim,value_dim, hidden_dim, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim,hidden_dim)
        self.key_proj = nn.Linear(key_dim,hidden_dim)
        self.value_proj = nn.Linear(value_dim,hidden_dim) #未使用过，先保持原样
        self.v = nn.Parameter(torch.Tensor(1,1,hidden_dim))
        self.b = nn.Parameter(torch.zeros(1,1,hidden_dim))
        stdv = 1./math.sqrt(self.v.size(2)) #初始化
        self.v.data.uniform_(-stdv,stdv)

    def forward(self, Q, K, V,mask=None):
        #mask:(bs, s_l)
        #Q[bs,node_num,dim]  K[bs,s_l,dim]  V[bs,s_l,dim]
        query = self.query_proj(Q) #[bs,node_num,hidden_dim]
        key = self.key_proj(K) #[bs,s_l,hidden_dim]

        query_expanded = query.unsqueeze(2).expand(-1, -1,K.size(1), -1)
        key_expanded = key.unsqueeze(1).expand(-1, Q.size(1),-1, -1)
        energies = torch.sum(self.v * torch.tanh(query_expanded + key_expanded +self.b), dim = -1)

        if mask!= None:
            mask = mask.unsqueeze(1).expand_as(energies)
            energies = energies.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(energies, dim=-1)
        context = torch.bmm(attn_weights,V)  # (batch_size, node_num, value_dim)

        return context, attn_weights


class PromptEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(PromptEmbedding, self).__init__()

        self.sent_dim = output_dim
        self.node_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = output_dim*2
        self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                 nn.GELU(),
                                 nn.Linear(input_dim//2, output_dim)
                                 )
        self.fc = nn.Linear(self.hidden_dim, self.sent_dim)
        self.additive_attention = AdditiveAttention(output_dim,output_dim,output_dim,output_dim)

        self.domain_projector = nn.Sequential(
            nn.Linear(output_dim,input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2,input_dim)
        )

        self.self_attn_layer = selfAttention_prompt(d_model=output_dim,num_head=4,dropout=dropout)

    def filter_Pool(self, x, mask):
        mask_expanded = mask.unsqueeze(2).type_as(x)
        mask_features = x * mask_expanded
        sum_features = torch.sum(mask_features, dim=1) # [batch_size, node_dim]
        #计算每个样本的有效节点数
        valid_node_num = torch.sum(mask, dim=1, keepdim=True).type_as(x)
        eps = torch.finfo(torch.float16).eps  # 获取float16的最小正值
        valid_node_counts = torch.clamp(valid_node_num, min=eps)
        average_data = sum_features / valid_node_counts
        return average_data

    def forward(self, text_embeds, node_embeds, gnn_mask, lm_mask):
        self.batch_size = gnn_mask.size(0)
        node_embeds = node_embeds.view(self.batch_size, -1 ,self.sent_dim)#[node_num,batch_size,sent_dim]因为不支持batch_first

        #text_prime = torch.mean(text_embeds, dim=1) # [batch_size, input_dim]

        H2 = self.self_attn_layer(node_embeds, mask = gnn_mask) #[batch_size, node_num, sent_dim]

        H3,_ = self.additive_attention(H2, text_embeds, text_embeds, mask = lm_mask)

        H4 = self.filter_Pool(H3, gnn_mask) #[batch_size, sent_dim][10,200]

        Z = self.domain_projector(H4)

        return Z

class PromptEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(PromptEmbedding, self).__init__()

        self.sent_dim = output_dim
        self.node_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = output_dim*2
        self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                 nn.GELU(),
                                 nn.Linear(input_dim//2, output_dim)
                                 )
        self.fc = nn.Linear(self.hidden_dim, self.sent_dim)
        self.additive_attention = AdditiveAttention(output_dim,output_dim,output_dim,output_dim)

        self.domain_projector = nn.Sequential(
            nn.Linear(output_dim,input_dim//2),
            nn.GELU(),
            nn.Linear(input_dim//2,input_dim)
        )

        self.self_attn_layer = selfAttention_prompt(d_model=output_dim,num_head=4,dropout=dropout)

    def filter_Pool(self, x, mask):
        mask_expanded = mask.unsqueeze(2).type_as(x)
        mask_features = x * mask_expanded
        sum_features = torch.sum(mask_features, dim=1) # [batch_size, node_dim]
        #计算每个样本的有效节点数
        valid_node_num = torch.sum(mask, dim=1, keepdim=True).type_as(x)
        eps = torch.finfo(torch.float16).eps  # 获取float16的最小正值
        valid_node_counts = torch.clamp(valid_node_num, min=eps)
        average_data = sum_features / valid_node_counts
        return average_data

    def forward(self, text_embeds, node_embeds, gnn_mask, lm_mask):
        self.batch_size = gnn_mask.size(0)
        node_embeds = node_embeds.view(self.batch_size, -1 ,self.sent_dim)#[node_num,batch_size,sent_dim]因为不支持batch_first

        #text_prime = torch.mean(text_embeds, dim=1) # [batch_size, input_dim]

        H2 = self.self_attn_layer(node_embeds, mask = gnn_mask) #[batch_size, node_num, sent_dim]

        H3,_ = self.additive_attention(H2, text_embeds, text_embeds, mask = lm_mask)

        H4 = self.filter_Pool(H3, gnn_mask) #[batch_size, sent_dim][10,200]

        Z = self.domain_projector(H4)

        return Z

class Text2graph(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.2): #0.1变为0.2
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
        self.norm = nn.LayerNorm(d_model)

    # def initialize(self):
    #     nn.init.kaiming_uniform_(self.W_Q.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_Q.bias)
    #     nn.init.kaiming_uniform_(self.W_K.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_K.bias)
    #     nn.init.kaiming_uniform_(self.W_V.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_V.bias)
    #     nn.init.kaiming_uniform_(self.W_O.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_O.bias)

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

    def forward(self, Q, K, V, node_num, mask=None):
        Q_copy = Q.clone()
        self.len_q = node_num
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
            alpha = F.softmax(A.masked_fill(_mask == 0, float('-inf')), dim=2)
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        out = self.norm(out)
        return out


class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_factor):
        super().__init__()
        self.input_dim = input_dim
        self.down_sample = nn.Linear(input_dim, input_dim // reduction_factor)
        self.activation = GELU()
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

    # def initialize(self):
    #     nn.init.kaiming_uniform_(self.W_Q.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_Q.bias)
    #     nn.init.kaiming_uniform_(self.W_K.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_K.bias)
    #     nn.init.kaiming_uniform_(self.W_V.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_V.bias)
    #     nn.init.kaiming_uniform_(self.W_O.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_O.bias)

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

    def forward(self, Q, K, V, node_num, mask=None):
        Q_copy = Q.clone()
        self.len_q = node_num
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
        out = self.adapter2(out)
        out = self.dropout2(out) + out_copy
        out = self.norm2(out)
        return out


class Graph2text(nn.Module):
    def __init__(self, h:int, d_model: int, len_q: int, len_k: int, d_k: int, d_v: int, dropout=0.2):#0.1变为0.2
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
        self.norm = nn.LayerNorm(d_model)


    # def initialize(self):
    #     nn.init.kaiming_uniform_(self.W_Q.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_Q.bias)
    #     nn.init.kaiming_uniform_(self.W_K.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_K.bias)
    #     nn.init.kaiming_uniform_(self.W_V.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_V.bias)
    #     nn.init.kaiming_uniform_(self.W_O.weight, nonlinearity='relu')
    #     nn.init.zeros_(self.W_O.bias)

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

    def forward(self, Q, K, V, node_num, mask=None, return_attn=False):
        Q_copy = Q.clone()
        batch_size = Q.size(0)
        self.len_k = node_num
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
                              dim=2)  # 在K的维度上进行mask，[batch_size*h,len_q,len_k]
        else:
            alpha = F.softmax(A, dim=2)  # [batch_size * h, len_q, len_k]
        out = torch.bmm(alpha, V).view([batch_size, self.h, self.len_q, self.d_v])  # [batch_size, h, len_q, d_v]
        out = out.permute([0, 2, 1, 3]).contiguous().view(
            [batch_size, self.len_q, self.out_dim])  # [batch_size, len_q, h * d_v]
        out = self.dropout(self.W_O(out)) + Q_copy
        out = self.norm(out)
        if return_attn:
            alpha = alpha.view(self.h,batch_size,self.len_q,self.len_k)
            alpha = alpha.mean(dim=0)
            alpha = alpha.permute(0,2,1)
            return out, alpha #（10，200，100）
        else:
            return out

class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


def run_test():
    print('testing BilinearAttentionLayer...')
    att = BilinearAttentionLayer(100, 20)
    mask = (torch.randn(70, 30) > 0).float()
    mask.requires_grad_()
    v = torch.randn(70, 30, 20)
    q = torch.randn(70, 100)
    o, _ = att(q, v, mask)
    o.sum().backward()
    print(mask.grad)

    print('testing DiffTopK...')
    x = torch.randn(5, 3)
    x.requires_grad_()
    k = 2
    r = DiffTopK.apply(x, k)
    loss = (r ** 2).sum()
    loss.backward()
    assert (x.grad == r * 2).all()
    print('pass')

    a = TripleEncoder()

    triple_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
    res = a(triple_input)
    print(res.size())

    b = LSTMEncoder(pooling=False)
    lstm_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    lengths = torch.tensor([3, 2])
    res = b(lstm_inputs, lengths)
    print(res.size())
