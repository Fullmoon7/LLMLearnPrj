# LLM Learning Repository

This repository contains my explorations and implementations related to Large Language Models (LLMs).

## Contents

* **`Tokenization/`**: Contains files related to tokenization techniques.
    * **`data/`**: Contains datasets used for training and testing tokenization methods.
        * `test.txt`: Sample testing data.
        * `test_BPE.txt`: Sample testing data for custom BPE and BBPE implementations.
        * `train.txt`: Sample training data.
        * `train_BPE.txt`: Sample training data for custom BPE and BBPE implementations.
    * `BBPE.py`: A custom implementation of the Byte-Based BPE (BBPE) algorithm.
    * `BPE.py`: A custom implementation of the Byte Pair Encoding (BPE) algorithm.
    * `tokenizers_BBPE.py`: BBPE implementation using the `tokenizers` library from Hugging Face.
    * `tokenizers_BPE.py`: BPE implementation using the `tokenizers` library from Hugging Face.

This repository will be updated with further explorations of LLM-related concepts.