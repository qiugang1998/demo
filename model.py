import torch
import torch.nn as nn
import numpy as np
import math
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        # vocab: 词表大小
        # dim: 词嵌入的维度
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, dim)
        
    def forward(self, x):
        return self.embedding(x) 
    
    
    
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=1000):
        # dim: 词嵌入的维度
        # max_len: 每个句子中包含词的最大个数
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 初始化位置编码矩阵
        P = torch.zeros(max_len, dim)

        # 绝对位置矩阵， 形状为 (max_len, 1), 对应论文中的 pos
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)

        # 每个词嵌入维度的偶数列， 对应公式中的 2i
        position_2i = torch.arange(0, dim, 2, dtype=torch.float32)

        # 传递给 sin 和 cos 的变量
        temp = pos / torch.pow(10000, position_2i / dim)


        # 计算偶数列位置编码
        P[:, 0::2] = torch.sin(temp)
        # 计算奇数列位置编码
        P[:, 1::2] = torch.cos(temp)
        
        
        
        # 扩充P的形状为(1,max_len, dim)
        P = P.unsqueeze(0)

        #位置编码矩阵在训练中是固定不变的，不会随着优化器二更新，将其注册为模型的buffer，
        #注册为buffer后，在模型保存后再重新加载这个模型，这个位置编码和模型参数会一并加载进来
        self.register_buffer('P', P)

    def forward(self, x):
        # x: 文本序列的词嵌入表示

        x = x + self.P[:, :x.shape[1]]
        return self.dropout(x)
        
        
# 计算注意力机制
def attention(query, key, value, mask=None, dropout=None):
    # mask: 掩码张量
    # dropout: 是一个实例化Dropout对象
    
    # 词嵌入的维度
    d_k = query.size(-1)

    # 计算 Q*(K的转置) / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 对scores做一个掩码操作
        scores = scores.masked_fill(mask == True, -float('inf'))
        
    softmax_scores = nn.functional.softmax(scores, dim=-1)
    


    if dropout is not None:
        softmax_scores = dropout(softmax_scores)

    return torch.matmul(softmax_scores, value)


"""多头注意力层"""
class MultiHeadAttention(nn.Module):
    
    def __init__(self, head, embedding_dim, dropout=0.1, bias=False):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % head == 0

        # 每个头的向量维度d_k
        self.d_k = embedding_dim // head

        self.dropout = nn.Dropout(p=dropout)

        # 传入头数
        self.head = head
        
        # 需要学习的矩阵
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_o = nn.Linear(embedding_dim,embedding_dim, bias=bias)

        

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        

        

        query = self.W_q(query).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        

        
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        

        return self.W_o(x)
    
    
# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        # hidden_dim: 中间层的输出维度
        super(PositionwiseFeedForward, self).__init__()

        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x：上一层的输出

        return self.dense2(self.dropout(self.relu(self.dense1(x))))
        
        
# 规范化层
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
    
    
    
# 子层连接
class SublayerConnection(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(SublayerConnection, self).__init__()
        
        self.norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # 接收上一层或者子层的输入作为第一个参数
        # sublayer: 子层连接中的子层函数

        return x + self.dropout(sublayer(self.norm(x)))
    
    
# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, multi_head_self_attention, feed_forward, dropout):
        # embedding_dim: 词的嵌入维度
        # multi_head_attention: 多头自注意力层的实例化对象
        # feed_forward: 前馈全连接层的实例化对象
        super(EncoderLayer, self).__init__()
        
        self.multi_head_self_attention = multi_head_self_attention
        self.feed_forward = feed_forward
        self.embedding_dim = embedding_dim

        # 编码器中有两个子层连接结构
        self.sublayer1 = SublayerConnection(embedding_dim, dropout)
        self.sublayer2 = SublayerConnection(embedding_dim, dropout)
        

    def forward(self, x, mask):
        

        x = self.sublayer1(x, lambda x: self.multi_head_self_attention(x, x, x, mask))

        return self.sublayer2(x, self.feed_forward)
    
    
# 编码器的实现
class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        # encoder_layer：编码器层的实例化对象
        super(Encoder, self).__init__()
        
        # 深拷贝 N 个编码器层
        self.encoder_layers= clones(encoder_layer, N)
        
        # 初始化规范化层，用于编码器的后面
        self.norm = LayerNorm(encoder_layer.embedding_dim)

    def forward(self, x, mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return self.norm(x)
    
# 解码层
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, multi_head_self_attention, multi_head_attention, feed_forward, dropout):
        # embedding_dim：词嵌入维度
        # multi_head_self_attention：多头自注意力对象
        # multi_head_attention：多头注意力对象
        # feed_forward: 前馈全连接层对象
        super(DecoderLayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.multi_head_self_attention = multi_head_self_attention
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward

        self.sublayers = clones(SublayerConnection(embedding_dim, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # x: 上一层的输入
        # memory: 编码器的输出

        x = self.sublayers[0](x, lambda x: self.multi_head_self_attention(x, x, x, target_mask))
        
        x = self.sublayers[1](x, lambda x: self.multi_head_attention(x, memory, memory, source_mask))

        return self.sublayers[2](x, self.feed_forward)
    
    
    
# 解码器的实现
class Decoder(nn.Module):
    def __init__(self, decoder_layer, N):
        # encoder_layer：解码器层的实例化对象
        super(Decoder, self).__init__()
        
        # 深拷贝 N 个解码器层
        self.decoder_layers= clones(decoder_layer, N)
        
        # 初始化规范化层，用于解码器的后面
        self.norm = LayerNorm(decoder_layer.embedding_dim)

    def forward(self, x, memory, source_mask, target_mask):
        # memory: 编码器的输出
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, source_mask, target_mask)
        return self.norm(x)

    
# 输出处理
class OutPut(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        # embedding_dim: 词嵌入维度
        # vocab_size: 词表大小
        super(OutPut, self).__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return nn.functional.softmax(self.linear(x), dim=-1)
    
# 编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, output):
        # encoder: 编码器对象
        # decoder：解码器对象
        # source_embed: 源数据的嵌入函数
        # target_embed: 目标数据的嵌入函数
        # output: 对解码器输出进行处理的对象
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.output = output

    def forward(self, source, target, source_mask, target_mask):
        memory = self.encoder(self.source_embed(source), source_mask)
        x = self.decoder(self.target_embed(target), memory, source_mask, target_mask)
        return  self.output(x)
    
    
# Transformer 模型
def TransformerModel(source_vocab, target_vocab, N=6, embedding_dim=512, hidden_dim=2048, head=8, dropout=0.1):
    # source_vocab: 源数据特征总数(词汇总数）
    # target_vocab: 目标数据特征总数(词汇总数）
    # N: 编码器解码器堆叠数
    # embedding_dim: 词嵌入维度
    # hidden_dim: 编码器中前馈全连接网络第一层的输出维度
    # head: 多头注意力结构中的多头数
    # dropout: 置零比率

    # 实例化多头注意力
    multi_head_attention = MultiHeadAttention(head, embedding_dim, dropout)

    # 实例化前馈全连接层
    feed_forward = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout)

    # 实例化位置编码类
    positional_encoding = PositionalEncoding(embedding_dim, dropout)

    # 实例化 EncoderDecoder 对象
    # 编码器：一个 attention + 一个前馈全连接
    # 解码器： ；两个 attention + 一个前馈全连接
    model = EncoderDecoder(
        Encoder(EncoderLayer(embedding_dim, copy.deepcopy(multi_head_attention), copy.deepcopy(feed_forward), dropout), N),
        Decoder(DecoderLayer(embedding_dim, copy.deepcopy(multi_head_attention), copy.deepcopy(multi_head_attention), copy.deepcopy(feed_forward), dropout), N),
        nn.Sequential(Embeddings(source_vocab, embedding_dim), copy.deepcopy(positional_encoding)), 
        nn.Sequential(Embeddings(target_vocab, embedding_dim), copy.deepcopy(positional_encoding)),
        OutPut(embedding_dim, target_vocab) 
    )

    # 初始化模型参数, 如果参数的维度大于 1，初始化为一个服从均匀分布的矩阵
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform(parameter)
        
    return model
