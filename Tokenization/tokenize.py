from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers

from tokenizers.trainers import BpeTrainer


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


train_data = load_data_from_file("data/train.txt")
test_data = load_data_from_file("data/test.txt")

if not train_data or not test_data:
    print("请确保train.txt和test.txt存在并包含数据。")
    exit()


# 初始化 BPE 模型
bpe_tokenizer = Tokenizer(BPE())


# 使用空格预分词器
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 初始化 BPE 训练器
trainer = BpeTrainer(vocab_size=5000, min_frequency=1, special_tokens=["[UNK]", "[CLS]",
                                                                     "[SEP]", "[PAD]",
                                                                     "[MASK]"]
                     ,show_progress=True)

# 训练分词器
bpe_tokenizer.train_from_iterator(train_data, trainer=trainer)

# 获取词汇表
vocabulary = bpe_tokenizer.get_vocab()
print("训练后的词汇表:", vocabulary)

# 测试分词器
print("\n测试集 Tokenization 结果:")
for text in test_data:
    output = bpe_tokenizer.encode(text)
    print(f"原始文本: {text}")
    print(f"Tokens ID: {output.ids}")
    print(f"Tokens: {output.tokens}")
    print("-" * 20)
