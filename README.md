# GPT-keras

## **GPT Keras implementation from scratch.**

This project is a TensorFlow implementation, based of Andrej Karpathy's video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) project, which can be found at the original link: [ng-video-lecture](https://github.com/karpathy/ng-video-lecture).

## About the Project

Andrej Karpathy's invaluable resource, guiding the process of coding GPT from the ground up and enabling it to generate text reminiscent of Shakespeare, serves as an excellent learning tool for delving into natural language processing and machine learning models. This TensorFlow implementation aims to replicate the functionality of the original project while utilizing the TensorFlow framework. This project uses tensorflow 2.10.1.

## Files

- [embedding_tf.py](examples/embedding_tf.py): simple implementation of Keras Embedding layer, that explained functionality how token-indices works with logits on operational level. Example is ready to use in any code.

- **bigram.py**: A Keras code that includes code for training a basic Bigram model. This model is a simple n-gram language model that can generate text by prediction the next token.

- **gpt.py**: A Keras code that includes the code for building a GPT model from scratch. It covers the implementation of a basic transformer architecture for text generation. You can follow the notebook to understand the inner workings of GPT.

- **gpt_model_weights.h5**: Pre-trained weights for the GPT model after 5,000 iterations. You can use these weights to generate text using the implemented GPT model without going through the training process.

- **input.txt**: This file includes the Shakespear work in a single file. It serves as the input for training both the Bigram model and the GPT model.

- **more.txt**: This file includes additional text data so that you can train the gpt on more data.

## Explanation of Code

The code in this repository is thoroughly explained in the accompanying code, that provides detailed insights into the implementation of the Generative Pre-trained Transformer (GPT) model from scratch using TensorFlow and Keras.


## References:

* https://github.com/j-planet/andrej_chatgpt
