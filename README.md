# Automatic Speech Recognition (ASR)
Converting Speech signals to Text format
Automatic speech recognition (ASR) converts a speech signal to text, mapping a sequence of audio inputs to text outputs. Virtual assistants like Siri and Alexa use ASR models to help users everyday, and there are many other useful user-facing applications like live captioning and note-taking during meetings.
In this project finetune Wav2Vec2 on the MInDS-14 dataset to transcribe audio to text.
Use your finetuned model for inference.
# Wav2Vec2 Model for Automatic Speech Recognition (ASR)<br>
The paper Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations described the facebook Wav2Vec2 model. Wav2Vec2 is a state-of-the-art deep learning model for automatic speech recognition (ASR). It is a type of self-supervised representation learning method that aims to learn a compact and highly discriminative representation of speech signals. The model is trained on large amounts of raw audio data without any manual transcriptions or alignment information, making it scalable and efficient. Wav2Vec2 has shown to significantly improve the performance of downstream ASR tasks, such as transcribing audio recordings into text, by providing better speech feature representations. The following figures shows the architecture of Wav2Vec2 model.
![image](https://user-images.githubusercontent.com/102625347/217045620-66360f7c-50e6-4343-b1a6-1ec64613432a.png)<br>
![image](https://user-images.githubusercontent.com/102625347/217045685-4e250ef2-e67f-47e4-8052-7633f8d585d4.png)<br>
The wav2vec 2.0 enabled speech recognition models achieved SOTA performance with a word error rate (WER) of 8.6 percent on noisy speech and 5.2 percent on clean speech on the standard LibriSpeech benchmark.<br>
# The Metric: word error rate (WER)
Word Error Rate (WER) is a common evaluation metric used in speech recognition and natural language processing to measure the accuracy of a generated transcript or translation compared to a reference text. It is defined as the ratio of the number of substitution, deletion, and insertion errors in the generated text to the number of words in the reference text. Lower WER values indicate a higher accuracy of the generated text compared to the reference.<br>
In this project, we will give an in-detail explanation of how Wav2Vec2's pretrained checkpoints can be fine-tuned on 'the minds' English ASR dataset.<br>
# Connectionist Temporal Classification (CTC) Algorithm
Connectionist Temporal Classification (CTC) is an algorithm used in deep learning models for speech recognition and natural language processing tasks. It is a form of loss function that trains a neural network to predict sequences of labels, such as phonemes or characters, from input sequences, such as audio signals or speech features.<br>

CTC is designed to handle input sequences of varying lengths, which is a common problem in speech recognition and natural language processing, where the length of the input sequence (e.g. length of an audio clip) may not be the same as the length of the output sequence (e.g. length of the transcription). The CTC loss function allows the network to learn an alignment between the input and output sequences, so that the network can produce correct transcriptions even when the lengths are different.<br>

CTC is implemented as a layer in a neural network architecture, and is trained end-to-end with the rest of the network. The output of the network is a probability distribution over the set of possible labels at each time step, and the CTC loss function computes the negative log-likelihood of the target transcription given the network's output probabilities. The network is trained to minimize the CTC loss, which results in improved recognition accuracy.<br>
# The "minds14" dataset
The "PolyAI/minds14" dataset is a speech recognition dataset that is part of the OpenAI Datasets collection. It is based on the "MInDS" (Mobile Integrated Diagnostic System) dataset and consists of spoken English phrases collected from a diverse set of speakers. The dataset includes both "train" and "test" splits and provides audio recordings and transcriptions of the phrases.

The "minds14" dataset is used for speech recognition research and can be used to train and evaluate machine learning models for this task. It provides a large and diverse set of speech data, which is important for training models that can generalize well to real-world scenarios. The dataset is also publicly available and well-documented, which makes it easy for researchers to access and use.

# Wav2Vec2CTCTokenizer<br>
The Wav2Vec2CTCTokenizer is a class from the Hugging Face Transformers library that is used to tokenize audio sequences into subword tokens for use with the Wav2Vec2 model.

The Wav2Vec2 model is a pre-trained deep neural network designed to handle raw audio waveform data, and the Wav2Vec2CTCTokenizer is used to convert the audio sequences into a format that the model can process. The tokenizer is based on the Connectionist Temporal Classification (CTC) loss function, which is commonly used in speech recognition.

The Wav2Vec2CTCTokenizer class takes a vocabulary file as input and uses it to map the subword tokens in the audio sequences to numerical values that can be processed by the Wav2Vec2 model. The vocabulary file is typically created by training a language model on a large corpus of text data.<br>

# Wav2Vec2FeatureExtractor <br>

Wav2Vec2FeatureExtractor is a class from the Hugging Face Transformers library that is used to extract features from audio sequences using the Wav2Vec2 model. The Wav2Vec2 model is a pre-trained deep neural network that is designed to handle raw audio waveform data.

The Wav2Vec2FeatureExtractor class is used to extract features from audio sequences that can then be used as input to other models, such as language models, for further processing. The extracted features represent a compact and meaningful representation of the audio data, and can be used for tasks such as speech recognition or audio classification.
# Wav2Vec2Processor <br>

Wav2Vec2Processor is a class from the Hugging Face Transformers library that is used to process audio sequences for use with the Wav2Vec2 model.<br>

The Wav2Vec2 model is a pre-trained deep neural network designed to handle raw audio waveform data, and the Wav2Vec2Processor class is used to convert the audio sequences into a format that the model can process.<br>

The Wav2Vec2Processor class combines the functionality of the Wav2Vec2CTCTokenizer and Wav2Vec2FeatureExtractor classes and provides a simple way to process audio sequences for use with the Wav2Vec2 model. The Wav2Vec2Processor takes an audio sequence as input, tokenizes it into subword tokens using the Wav2Vec2CTCTokenizer, and then extracts features from the tokenized sequence using the Wav2Vec2FeatureExtractor.<br>

The Wav2Vec2Processor class makes it easy to preprocess audio sequences for use with the Wav2Vec2 model, and eliminates the need to manually perform the tokenization and feature extraction steps.<br>
