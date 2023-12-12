# transformers

# transformer
1. 输入编码（Embedding），将输入序列（例如，文本中的单词或句子）的每个元素（单词或 token）映射为高维空间中的向量。
2. 位置编码（Positional Encoding）
3. 多头注意力机制 Attention(Q, K, V) = softmax(Q*Kt/sqrt(dk))*V 计算相似度
4. 前馈神经网络，通过全连接层和激活函数，对每个位置的表示进行非线性变换

# text to text
1. 将输入文本（text）通过 tokenizer 进行编码，生成 token。
2. 使用 Transformer 模型的 encoder 将 token 转换为张量。
3. 对张量进行 softmax 操作，得到概率分布。
4. 选择概率最大的值，得到预测的 token。
5. 将预测的 token 通过 tokenizer 的 decoder 进行解码，生成最终的预测文本。

