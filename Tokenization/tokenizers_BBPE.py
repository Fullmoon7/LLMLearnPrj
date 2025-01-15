from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers

from tokenizers.trainers import BpeTrainer


# 从文件夹中加载数据并转换为字节
def load_data_as_bytes(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip().encode('utf-8'))  # 将每行文本编码为字节
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    return data


train_data_bytes = load_data_as_bytes("data/train.txt")
test_data_bytes = load_data_as_bytes("data/test.txt")

if not train_data_bytes or not test_data_bytes:
    print("请确保train.txt和test.txt存在并包含数据。")
    exit()

# 初始化 BPE 模型
bbpe_tokenizer = Tokenizer(BPE())

# **关键修改:** BBPE 不需要预分词器，因为最小单位已经是字节
# bbpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # 删除这行

# 初始化 BPE 训练器
trainer = BpeTrainer(vocab_size=5000, min_frequency=1, special_tokens=["[UNK]", "[CLS]",
                                                                     "[SEP]", "[PAD]",
                                                                     "[MASK]"]
                     ,show_progress=True)


# **关键修改: 提供一个生成器，将每个字节序列分解为单独的字节**
def byte_iterator(data):
    for byte_sequence in data:
        for byte in byte_sequence:  # 遍历字节序列中的每个字节
            yield chr(byte)  # 将每个字节转换为 bytes 对象并 yield


# 训练分词器
bbpe_tokenizer.train_from_iterator(byte_iterator(train_data_bytes), trainer=trainer)


# 获取词汇表
vocabulary = bbpe_tokenizer.get_vocab()
print("训练后的词汇表:", vocabulary)

# 测试分词器
print("\n测试集 Byte-BPE Tokenization 结果:")
for text_bytes in test_data_bytes:
    # **关键修改:** 将字节序列转换为字符串
    text_as_string = "".join([chr(byte) for byte in text_bytes])
    output = bbpe_tokenizer.encode(text_as_string)
    decoded_text = text_bytes.decode('utf-8', errors='replace')  # 为了方便显示，将字节解码回文本
    print(f"原始文本(Bytes): {text_bytes}")
    print(f"原始文本(Decoded): {decoded_text}")
    print(f"Tokens ID: {output.ids}")
    print(f"Tokens: {output.tokens}")
    print("-" * 20)
