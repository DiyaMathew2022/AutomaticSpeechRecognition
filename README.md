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
## The Metric: word error rate (WER)
Word Error Rate (WER) is a common evaluation metric used in speech recognition and natural language processing to measure the accuracy of a generated transcript or translation compared to a reference text. It is defined as the ratio of the number of substitution, deletion, and insertion errors in the generated text to the number of words in the reference text. Lower WER values indicate a higher accuracy of the generated text compared to the reference.<br>
In this project, we will give an in-detail explanation of how Wav2Vec2's pretrained checkpoints can be fine-tuned on 'the minds' English ASR dataset.<br>
# Connectionist Temporal Classification (CTC)
Connectionist Temporal Classification (CTC) is an algorithm used in deep learning models for speech recognition and natural language processing tasks. It is a form of loss function that trains a neural network to predict sequences of labels, such as phonemes or characters, from input sequences, such as audio signals or speech features.<br>

CTC is designed to handle input sequences of varying lengths, which is a common problem in speech recognition and natural language processing, where the length of the input sequence (e.g. length of an audio clip) may not be the same as the length of the output sequence (e.g. length of the transcription). The CTC loss function allows the network to learn an alignment between the input and output sequences, so that the network can produce correct transcriptions even when the lengths are different.<br>

CTC is implemented as a layer in a neural network architecture, and is trained end-to-end with the rest of the network. The output of the network is a probability distribution over the set of possible labels at each time step, and the CTC loss function computes the negative log-likelihood of the target transcription given the network's output probabilities. The network is trained to minimize the CTC loss, which results in improved recognition accuracy.<br>
