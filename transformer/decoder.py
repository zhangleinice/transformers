import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from transformers import AutoConfig, AutoTokenizer
from encoder import Embeddings, FeedForward, TransformerEncoder

class DecoderAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, output_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.output_linear = nn.Linear(head_dim, output_dim)

    def forward(self, query, key, value, mask=None):
        query, key, value = self.q(query), self.k(key), self.v(value)

        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)  # 处理NaN值
        output = torch.bmm(weights, value)
        return self.output_linear(output)

class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [DecoderAttentionHead(embed_dim, head_dim, output_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(num_heads * output_dim, embed_dim)

    def forward(self, query, key, value, mask=None, query_mask=None, key_mask=None):
        if query_mask is not None and key_mask is not None:
            mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))

        outputs = [h(query, key, value, mask) for h in self.heads]
        x = torch.cat(outputs, dim=-1)
        x = self.output_linear(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        output_dim = config.hidden_size
        self.self_attention = DecoderMultiHeadAttention(config, output_dim)
        self.cross_attention = DecoderMultiHeadAttention(config, output_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        hidden_state = self.layer_norm_1(x)
        self_attention_output = x + self.self_attention(hidden_state, hidden_state, hidden_state, mask=self_mask)
        cross_attention_output = self_attention_output + self.cross_attention(self_attention_output, encoder_output, encoder_output, mask=cross_mask)
        x = cross_attention_output + self.feed_forward(self.layer_norm_2(cross_attention_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return x
    
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def generate_cross_mask(input_ids, max_length):
    mask = torch.ones((max_length, max_length))
    mask = torch.tril(mask)
    return mask

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    print('inputs', inputs)

    # 编码器
    encoder = TransformerEncoder(config)

    print('input_ids', encoder(inputs.input_ids))
    print('input_size', encoder(inputs.input_ids).size())

    encoder_output = encoder(inputs.input_ids)
    
    # 解码器
    decoder = TransformerDecoder(config)

    max_length = inputs.input_ids.size(1)

    self_mask = decoder.generate_square_subsequent_mask(max_length).to(device)
    cross_mask = generate_cross_mask(inputs.input_ids, max_length).to(device)

    decoded_output = decoder(inputs.input_ids, encoder_output, self_mask, cross_mask)
    
    print('decoded_output', decoded_output)

    # softmax 归一化
    generator = nn.Linear(config.hidden_size, tokenizer.vocab_size)
    logits = generator(decoded_output)
    probabilities = F.softmax(logits, dim=-1)

    # 生成概率分布
    print('Probabilities:', probabilities)

    # 预测下一个词
    _, predicted_indices = torch.max(probabilities, dim=-1)

    # 选择第一个位置的预测词
    predicted_index = predicted_indices[0, 0].item()

    # 使用tokenizer.decode将索引转换为文本
    predicted_word = tokenizer.decode(predicted_index)

    print('Predicted Word:', predicted_word)


