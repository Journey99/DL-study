# transformer 의 핵심 구조 구현

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.size()
        q = self.q_linear(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (B, h, T, d_k)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x2 = self.norm1(x + self.attn(x, x, x, mask))
        x3 = self.norm2(x2 + self.ff(x2))
        return x3

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x2 = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x3 = self.norm2(x2 + self.cross_attn(x2, enc_out, enc_out, src_mask))
        x4 = self.norm3(x3 + self.ff(x3))
        return x4


class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, max_len=100):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model) # 입력 단어 벡터화
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model) 
        self.pos_enc = PositionalEncoding(d_model, max_len) # 위치 정보

        self.encoder = EncoderLayer(d_model, num_heads, d_ff) # 입력 문장을 압축된 벡터로 변환
        self.decoder = DecoderLayer(d_model, num_heads, d_ff) # 벡터로 출력 문장 생성

        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_enc(self.src_embed(src))
        tgt = self.pos_enc(self.tgt_embed(tgt))

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)

        return self.out(dec_out)
    


# 예제 사용
src = torch.randint(0, 1000, (2, 10))  # (batch, src_seq_len)
tgt = torch.randint(0, 1000, (2, 10))  # (batch, tgt_seq_len)

model = SimpleTransformer(src_vocab_size=1000, tgt_vocab_size=1000)
output = model(src, tgt)

print(model)
print(output.shape)  # (batch_size, tgt_seq_len, tgt_vocab_size)
