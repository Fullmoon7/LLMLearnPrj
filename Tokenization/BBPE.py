import re
from collections import defaultdict

class BBPE_Hex:
    def __init__(self, vocab_size, min_frequency=2, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = {token: i for i, token in enumerate(special_tokens)}
        self.vocab = self.special_tokens.copy()
        self.merges = {}

    def _bytes_to_hex(self, byte_seq):
        return ''.join([f'{b:02x}' for b in byte_seq])

    def _hex_to_bytes(self, hex_str):
        return bytes.fromhex(hex_str)

    def _build_vocab(self, corpus):
        word_freqs = defaultdict(int)
        for sentence in corpus:
            words = sentence.strip().split()
            for word in words:
                byte_word = ' '.join([f'{b:02x}' for b in word.encode('utf-8')])
                word_freqs[byte_word] += 1
        initial_vocab = set()
        for hex_word in word_freqs.keys():
            for hex_byte in hex_word.split():
                initial_vocab.add(hex_byte)

        for hex_byte in sorted(list(initial_vocab)):
            if hex_byte not in self.vocab:
                self.vocab[hex_byte] = len(self.vocab)

        return word_freqs

    def _get_stats(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            hex_bytes = word.split()
            for i in range(len(hex_bytes) - 1):
                pairs[tuple(hex_bytes[i:i + 2])] += freq
        return pairs

    def _merge_vocab(self, pair, word_freqs):
        new_word_freqs = {}
        bigram_hex = ' '.join(pair)
        replacement_hex = ''.join(pair)  # 合并后的十六进制表示

        p = re.compile(rf'(?<!\S){re.escape(bigram_hex)}(?!\S)')
        for word, freq in word_freqs.items():
            new_word = p.sub(replacement_hex, word)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, corpus):
        print("构建初始词汇表...")
        word_freqs = self._build_vocab(corpus)
        print(f"初始词汇表大小: {len(self.vocab)}")

        num_merges = 0
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            max_pair = max(pairs, key=pairs.get)

            if pairs[max_pair] < self.min_frequency:
                print(f"最高频字节对 '{max_pair}' 的频率 ({pairs[max_pair]}) 低于最小频率阈值 {self.min_frequency}, 停止合并。")
                break

            word_freqs = self._merge_vocab(max_pair, word_freqs)
            self.merges[max_pair] = len(self.vocab)
            new_token_hex = "".join(max_pair)  # 合并后的十六进制字符串
            if new_token_hex not in self.vocab:
                self.vocab[new_token_hex] = len(self.vocab)

            # 解码合并后的 token 以便查看
            try:
                decoded_token = self._hex_to_bytes(new_token_hex).decode('utf-8')
            except UnicodeDecodeError:
                decoded_token = f"Bytes: {self._hex_to_bytes(new_token_hex)}"

            num_merges += 1
            print(f"合并: {max_pair} -> {new_token_hex} (Decoded: '{decoded_token}'), 新词汇表大小: {len(self.vocab)}")

        print(f"最终合并次数: {num_merges}")

    def _tokenize_word(self, word):
        byte_word = word.encode('utf-8')
        hex_word = self._bytes_to_hex(byte_word)
        tokens = []
        start = 0
        while start < len(hex_word):
            end = len(hex_word)
            best_token = "[UNK]"
            best_len = 1
            while start < end:
                potential_token = hex_word[start:end]
                if potential_token in self.vocab:
                    if len(potential_token) > best_len:
                        best_token = potential_token
                        best_len = len(potential_token)
                end -= 2  # 每次移动两个字符，因为是十六进制
                if end < start:
                    break

            if best_token != "[UNK]":
                tokens.append(best_token)
                start += len(best_token)
            else:
                # 如果没有找到匹配的 token，则将当前的两个十六进制字符作为一个 token (代表一个字节)
                if start + 2 <= len(hex_word):
                    tokens.append(hex_word[start:start+2])
                    start += 2
                else:
                    break # 避免越界
        return tokens

    def tokenize(self, text):
        preprocessed_text = text.lower()
        words = preprocessed_text.split()
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get("[UNK]")) for token in tokens]

    def decode_tokens(self, tokens):
        decoded_tokens = []
        for token in tokens:
            if token in self.special_tokens:
                decoded_tokens.append(token)
            elif all(c in '0123456789abcdef' for c in token):
                try:
                    decoded_tokens.append(self._hex_to_bytes(token).decode('utf-8', errors='replace'))
                except:
                    decoded_tokens.append("[UNK]")
            else:
                decoded_tokens.append("[UNK]")
        return decoded_tokens

    def decode(self, ids):
        reverse_vocab = {i: token for token, i in self.vocab.items()}
        tokens = [reverse_vocab.get(i, "[UNK]") for i in ids]
        decoded_tokens = self.decode_tokens(tokens)
        return " ".join(decoded_tokens)

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

# 初始化并训练 BBPE 模型
bbpe_hex = BBPE_Hex(vocab_size=5000, min_frequency=2)
bbpe_hex.train(train_data)

# 打印词汇表 (部分)
print("\n训练后的 BBPE (Hex) 词汇表 (部分):")
for i, (token, index) in enumerate(bbpe_hex.vocab.items()):
    print(f"{token}: {index}")
    if i >= 50:
        break

# 测试分词器
print("\n测试集 BBPE (Hex) Tokenization 结果:")
for text in test_data:
    tokens = bbpe_hex.tokenize(text)
    ids = bbpe_hex.encode(text)
    decoded_tokens = bbpe_hex.decode_tokens(tokens)
    print(f"原始文本: {text}")
    print(f"Tokens: {tokens}")
    print(f"Tokens ID: {ids}")
    print(f"Decoded Tokens: {decoded_tokens}")
    print(f"Decoded Text: {bbpe_hex.decode(ids)}")
    print("-" * 20)