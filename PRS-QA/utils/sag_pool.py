from typing import Union, Optional, Callable, Tuple
from torch import Tensor
import torch
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if :obj:`min_score` :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Callable = GraphConv, min_score: Optional[float] = None,
                 multiplier: float = 1.0, nonlinearity: Callable = torch.tanh,
                 **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        #self.max_node_num = 2000

    # def filter_edge_lengths(self, edge_index_pre_pool, edge_index_post_pool, edge_lengths, perm):
    #     # 创建一个最大节点索引+1长度的零张量，用于标记是否保留节点
    #     # node_mask = torch.zeros(self.max_node_num, dtype=torch.bool)
    #     # node_mask[perm] = True  # 标记保留的节点
    #
    #     #确定哪些边在池化前后都存在
    #     # 创建一个mask，初始为全False，长度与edge_index_pre_pool的第二维相同
    #     edge_mask = torch.zeros(edge_index_pre_pool.size(1), dtype=torch.bool)
    #
    #     #遍历池化后的边索引，缉拿查每条边是否在池化前的边索引中存在
    #     for i in range(edge_index_post_pool.size(1)):
    #         #池化之后的每条边的两个节点
    #         node_a,node_b = edge_index_post_pool[:,i]
    #         matches = (edge_index_pre_pool[0] == node_a) & (edge_index_pre_pool[1] == node_b)
    #         #将找到的边在edge_mask中标记为True
    #         edge_mask[matches] = True
    #
    #     # 使用edge_mask来筛选edge_lengths
    #     filtered_edge_lengths = edge_lengths[edge_mask]
    #     return filtered_edge_lengths

    def filter_adj(self,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            node_index: Tensor,
            cluster_index: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
            edge_lengths = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if cluster_index is None:
            cluster_index = torch.arange(node_index.size(0),
                                         device=node_index.device)

        mask = node_index.new_full((num_nodes,), -1)
        mask[node_index] = cluster_index

        row, col = edge_index[0], edge_index[1]
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        #同时筛选edge_lengths
        if edge_lengths is not None:
            edge_lengths = edge_lengths[mask]

        return torch.stack([row, col], dim=0), edge_attr, edge_lengths



    def forward(self, x, score, edge_index, edge_attr, node_type, batch=None, edge_lengths=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score) #选择top-k算法
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x #池化后的特征乘以的系数。这可以用于调整特征的规模。

        batch = batch[perm]

        node_type = node_type[perm]


        edge_index_post_pool, edge_attr_post_pool, new_edge_lengths = self.filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0),edge_lengths=edge_lengths)


        return x, edge_index_post_pool, edge_attr_post_pool, node_type, batch, perm, score[perm], new_edge_lengths


    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)