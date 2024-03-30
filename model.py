import torch
import torch.nn as nn
import numpy as np
import math
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
# 位置编码层
class PositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_len=1000):
        # embedding_dim：词嵌入维度
        # vocab_size: 所有训练的词汇数
        # max_len: 一个句子中最大的词汇数
        super(PositionEmbedding, self).__init__()
        # max_len: 每个句子中的最大词汇数
        
        # pos是第几个词,i是第几个维度, d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(max_len, embedding_dim)
        for i in range(max_len):
            for j in range(embedding_dim):
                pe[i, j] = get_pe(i, j, embedding_dim)
                
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        embed = self.embed(x)

        # 词编码和位置编码相加
        return embed + self.pe[:, :x.shape[1]]
        
        
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
        # embedding_dim：词嵌入维度
        # head：多头注意力中的头数
        
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

        self.norm = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        
        
        clone_query = query.clone()
        
        #试验证明先归一化效果更好
        query = self.norm(query)
        key = self.norm(key)
        value = self.norm(value)


        query = self.W_q(query).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        

        
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        
        # return self.norm(self.dropout(self.W_o(x)) + clone_query)
        return self.dropout(self.W_o(x)) + clone_query
    
    
# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        # hidden_dim: 中间层的输出维度
        super(PositionwiseFeedForward, self).__init__()

        self.dense1 = nn.Linear(embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)

    def forward(self, x):
        # x：上一层的输出
        
        clone_x = x.clone()
        
        #规范化
        x = self.norm(x)
        
        #return self.norm(self.dropout(self.dense2(self.relu(self.dense1(x)))) + clone_x)
        return self.dropout(self.dense2(self.relu(self.dense1(x)))) + clone_x
        
           
    
# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head, hidden_dim, dropout):
        # embedding_dim: 词的嵌入维度
        # head: 多头自注意力的头数
        # hidden_dim: 前馈全连接层中间层的输出维度
        
        
        super(EncoderLayer, self).__init__()
        
        
        # 多头自注意力层
        self.multi_head_self_attention = MultiHeadAttention(head, embedding_dim, dropout)
        
        
        #前馈全连接层
        self.feed_forward = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout)
     

    def forward(self, x, mask):
        

        x = self.multi_head_self_attention(x, x, x, mask)

        return self.feed_forward(x)
    
    
# 编码器的实现
class Encoder(nn.Module):
    def __init__(self, embedding_dim, head, hidden_dim, dropout, N):
        # embedding_dim：词嵌入维度
        # head: 多头注意力中的头
        # hidden_dim：前馈全连接层中间层的输出维度
        # N: 编码层的个数
        super(Encoder, self).__init__()
        
        temp_encoder_layer = EncoderLayer(embedding_dim, head, hidden_dim, dropout)
        
        # 深拷贝 N 个编码器层
        self.encoder_layers= clones(temp_encoder_layer, N)
        

    def forward(self, x, mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x
    
# 解码层
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, head, hidden_dim, dropout):
        # embedding_dim：词嵌入维度
        # head：多头自注意力层的头数
        # hidden_dim: 前馈全连接层中间层的输出维度
        
        super(DecoderLayer, self).__init__()
        

        self.multi_head_self_attention = MultiHeadAttention(head, embedding_dim, dropout)
        self.multi_head_attention = MultiHeadAttention(head, embedding_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout)


    def forward(self, x, y, source_mask, target_mask):
        # y: 上一层的输入
        # x: 编码器的输出

        y = self.multi_head_self_attention(y, y, y, target_mask)
        
        y = self.multi_head_attention(y, x, x, source_mask)
        
        

        return self.feed_forward(y)
    
    
    
# 解码器的实现
class Decoder(nn.Module):
    def __init__(self, embedding_dim, head, hidden_dim, dropout, N):
        
        super(Decoder, self).__init__()
        
        temp_decoder_layer = DecoderLayer(embedding_dim, head, hidden_dim, dropout)
        # 深拷贝 N 个解码器层
        self.decoder_layers= clones(temp_decoder_layer, N)
        

    def forward(self, x, y, source_mask, target_mask):
        # x: 编码器的输出
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(x, y, source_mask, target_mask)
        return y

    
# 输出处理
class OutPut(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        # embedding_dim: 词嵌入维度
        # vocab_size: 输出的词表大小
        super(OutPut, self).__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, y):
        return nn.functional.softmax(self.linear(y), dim=-1)
    
# 主模型
class Transformer(torch.nn.Module):
    def __init__(self, source_vocab, target_vocab, N, embedding_dim, hidden_dim, head, dropout):
        
        # source_vocab: 源数据特征总数(词汇总数）
        # target_vocab: 目标数据特征总数(词汇总数）
        # N: 编码器解码器堆叠数
        # embedding_dim: 词嵌入维度
        # hidden_dim: 编码器中前馈全连接网络第一层的输出维度
        # head: 多头注意力结构中的多头数
        # dropout: 置零比率

        
        super(Transformer, self).__init__()
        
        self.embed_x = PositionEmbedding(embedding_dim, source_vocab)
        self.embed_y = PositionEmbedding(embedding_dim, target_vocab)
        self.encoder = Encoder(embedding_dim, head, hidden_dim, dropout, N)
        self.decoder = Decoder(embedding_dim, head, hidden_dim, dropout, N)
        self.output = torch.nn.Linear(embedding_dim, target_vocab)

    def forward(self, x, y, mask_x, mask_y):


        x = self.embed_x(x)
        y = self.embed_y(y)

        # 编码器计算
        x = self.encoder(x, mask_x)

        # 解码器计算
        y = self.decoder(x, y, mask_x, mask_y)

        # 全连接输出
        y = self.output(y)

        return y
