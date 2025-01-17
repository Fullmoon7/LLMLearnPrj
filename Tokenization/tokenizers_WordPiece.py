from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase

# 1. 初始化 WordPiece 模型
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 2. 设置 Normalizer (可选，但推荐)
tokenizer.normalizer = NFD()
tokenizer.normalizer = Lowercase()

# 3. 设置 Pre-tokenizer (用于将文本分割成初始的词)
tokenizer.pre_tokenizer = Whitespace()

# 4. 初始化 WordPiece Trainer
trainer = WordPieceTrainer(
    vocab_size=30000,  # 设置词表大小
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  # 定义特殊 token
    min_frequency=2,  # 设置子词的最小出现频率
    limit_alphabet=1000 # 限制初始字符表大小，可以加快训练速度
)

# 5. 提供训练数据 (可以是文件列表或字符串列表)
training_data = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And this is another sentence.",
    "WordPiece is a subword tokenization algorithm.",
    "It helps to handle out-of-vocabulary words.",
    "The tokenizer library is easy to use."
]

# 如果训练数据在文件中，可以使用以下方式：
# training_files = ["train.txt", "val.txt"]

# 6. 训练 Tokenizer
tokenizer.train_from_iterator(training_data, trainer=trainer)
# 如果使用文件训练：
# tokenizer.train(files=training_files, trainer=trainer)

# 7. 使用 Tokenizer 进行编码
output = tokenizer.encode("This is a new sentence about WordPiece.")

print("Tokens:", output.tokens)
print("IDs:", output.ids)
print("Attention Mask:", output.attention_mask)

# 8. 解码回文本
decoded_text = tokenizer.decode(output.ids)
print("Decoded Text:", decoded_text)

# 9. 保存和加载 Tokenizer
tokenizer.save("wordpiece_tokenizer.json")
loaded_tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

# 10. 再次使用加载的 Tokenizer
output_loaded = loaded_tokenizer.encode("Let's try the loaded tokenizer.")
print("Loaded Tokenizer Tokens:", output_loaded.tokens)