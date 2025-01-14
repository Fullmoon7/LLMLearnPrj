import re
from collections import defaultdict


class BPE:
    def __init__(self, vocab_size, min_frequency=2, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.vocab = {}
        self.merges = {}

    def _build_vocab(self, corpus):
        """
        步骤 1: 构建初始词汇表
        """
        # 统计词频
        word_freqs = defaultdict(int)
        for sentence in corpus:
            words = sentence.strip().split()  # 简单按空格分词
            for word in words:
                word_freqs[' '.join(list(word))] += 1

        # 初始化词汇表，包括所有单个字符和特殊 token
        vocab = set()
        for word in word_freqs.keys():
            for char in word.split():
                vocab.add(char)
        for token in self.special_tokens:
            vocab.add(token)

        return word_freqs, {token: i for i, token in enumerate(sorted(list(vocab)))}

    def _get_stats(self, word_freqs):
        """
        步骤 2.1: 统计字节对/字符对频率
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(self, pair, word_freqs):
        """
        步骤 2.2: 合并最高频的字节对/字符对
        """
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freqs:
            w_out = p.sub(''.join(pair), word)
            new_word_freqs[w_out] = word_freqs[word]
        return new_word_freqs

    def train(self, corpus):
        """
        训练 BPE 模型
        """
        print("构建初始词汇表...")
        word_freqs, self.vocab = self._build_vocab(corpus)
        print(f"初始词汇表大小: {len(self.vocab)}")

        # 迭代合并
        num_merges = 0
        while len(self.vocab) < self.vocab_size:  # 检查当前词表大小是否大于预设
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            max_pair = max(pairs, key=pairs.get)

            # 检查合并频率是否低于阈值
            if pairs[max_pair] < self.min_frequency:
                print(f"最高频字节对 '{max_pair}' 的频率 ({pairs[max_pair]}) 低于最小频率阈值 {self.min_frequency}, 停止合并。")
                break

            word_freqs = self._merge_vocab(max_pair, word_freqs)
            self.merges[max_pair] = len(self.vocab)  # 记录合并顺序
            new_token = "".join(max_pair)
            self.vocab[new_token] = len(self.vocab)
            num_merges += 1
            print(f"合并: {max_pair} -> {new_token}, 新词汇表大小: {len(self.vocab)}")

        print(f"最终合并次数: {num_merges}")

    def _tokenize_word(self, word):
        """
        对单个单词进行分词
        """
        if word in self.vocab:
            return [word]

        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                # 说明词汇表中没有以start指向字符开头的词，则把当前字符记为一个[UNK]
                tokens.append("[UNK]")
                start += 1
            else:
                # 说明匹配到一个以start指向字符开头的最长子串，则添加到tokens列表中
                tokens.append(cur_substr)
                start = end  # 继续处理剩下的字符串
        return tokens

    def tokenize(self, text):
        """
        对文本进行分词
        """
        # 可以添加更复杂的预处理，例如标点符号分割等
        preprocessed_text = text.lower()
        words = preprocessed_text.split()
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        return tokens

    def encode(self, text):
        """
        将文本编码为 ID 序列
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

    def decode(self, ids):
        """
        将 ID 序列解码为文本
        """
        reverse_vocab = {i: token for token, i in self.vocab.items()}
        tokens = [reverse_vocab.get(i, "[UNK]") for i in ids]
        return " ".join(tokens)

# 从文件夹中加载数据
def load_data_from_file(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    return data

# 示例使用
train_data = load_data_from_file("data/train_BPE.txt")
test_data = load_data_from_file("data/test_BPE.txt")

if not train_data or not test_data:
    print("请确保train.txt和test.txt存在并包含数据。")
    exit()

# 初始化并训练 BPE 模型
bpe = BPE(vocab_size=5000, min_frequency=2)
bpe.train(train_data)

# 打印词汇表 (部分)
print("\n训练后的词汇表 (部分):")
for i, (token, index) in enumerate(bpe.vocab.items()):
    print(f"{token}: {index}")
    if i >= 20:
        break

# 测试分词器
print("\n测试集 Tokenization 结果:")
for text in test_data:
    tokens = bpe.tokenize(text)
    ids = bpe.encode(text)
    print(f"原始文本: {text}")
    print(f"Tokens: {tokens}")
    print(f"Tokens ID: {ids}")
    print("-" * 20)
