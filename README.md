# DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

We propose DistriBlock, a novel detection method for adversarial attacks on neural network-based ASR systems. We show that characteristics of the distribution over the output tokens can serve as features of binary classifiers.

![logo](resources/Detector_diagram_3.png)

## Prerequisites
We analyze a variety of fully integrated PyTorch-based deep learning E2E speech engines using [SpeechBrain](https://github.com/speechbrain/speechbrain). Please refer to their website for instructions on how to install it.
We perform evaluations of our detectors using an NVIDIA A40 GPU with 48 GB of memory, along with ASR recipes from SpeechBrain version 0.5.14.

### Pre-trained models
Speechbrain contains pre-trained models that can be used to generate adversarial examples and test our defense strategy:
* [CRDNN with CTC/Attention trained on CommonVoice Italian](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-it)
* [CRDNN with CTC/Attention trained on LibriSpeech](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech)
* [wav2vec 2.0 with CTC trained on Aishell](https://huggingface.co/speechbrain/asr-wav2vec2-ctc-aishell)
* [wav2vec 2.0 with CTC trained on CommonVoice German](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-de)
* [Transformer trained on Aishell](https://huggingface.co/speechbrain/asr-transformer-aishell)
* [Transformer trained on LibriSpeech](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)
  
### Datasets

### Adversarial attacks
To generate the Adversarial Examples, we utilized [RobustSpeech](https://github.com/RaphaelOlivier/robust_speech) repository that contains a PyTorch implementation of all considered attacks.

## Computing Characteristics

## Building binary Classifiers
