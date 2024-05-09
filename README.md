# DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

We propose DistriBlock, a novel detection method for adversarial attacks on neural network-based ASR systems. We show that characteristics of the distribution over the output tokens can serve as features of binary classifiers.

![logo](resources/Detector_diagram_3.png)

## Prerequisites
We analyze a variety of fully integrated PyTorch-based deep learning E2E speech engines using [SpeechBrain](https://github.com/speechbrain/speechbrain). Please refer to their website for instructions on how to install it.
We perform evaluations of our detectors using an NVIDIA A40 GPU with 48 GB of memory, along with ASR recipes from SpeechBrain version 0.5.14.

### Datasets

### Adversarial attacks
To generate the Adversarial Examples, we utilized [RobustSpeech](https://github.com/RaphaelOlivier/robust_speech) repository that contains a PyTorch implementation of all considered attacks.

### Pre-trained models

## Computing Characteristics

## Building binary Classifiers
