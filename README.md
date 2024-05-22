# [UAI 2024] DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

We propose DistriBlock, a novel detection method for adversarial attacks on neural network-based ASR systems. 
We show that characteristics of the distribution over the output tokens can serve as features of binary classifiers.

![logo](resources/Detector_diagram_3.png)

A demo with a selection of benign, adversarial, and noisy data employed in our experiments is available online [DistriBlock Demo](https://matiuste.github.io/Distriblock_demo/)
> Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat.
> To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step.
> We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. 
> Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. 
> Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99\% and 97\%, respectively. 
> To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

## Prerequisites
Before running the Distriblock scripts, the following tasked need to be completed:
1. SpeechBrain
2. Datasets
3. Pre-trained ASR models
4. Adversarial attacks
   
#### 1. SpeechBrain
We analyze a variety of fully integrated PyTorch-based deep learning E2E speech engines using [SpeechBrain](https://github.com/speechbrain/speechbrain). 
Please refer to their website for instructions on how to install it.
We perform evaluations of our detectors using an NVIDIA A40 GPU with 48 GB of memory, along with ASR recipes from SpeechBrain version 0.5.14.

#### 2. Datasets
We use the following large-scale speech corpus:
* [LibriSpeech (English)](https://www.openslr.org/12)
* [Aishell (Chinese Mandarin)](https://www.openslr.org/33/)
* [Common Voice 6.1 (German and Italian)](https://commonvoice.mozilla.org/en/datasets)

    .
    ├── build                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    ├── LICENSE
    └── README.md
  
#### 3. Pre-trained ASR models
Speechbrain contains pre-trained models that can be used to generate adversarial examples and to test our defense strategy:
* [CRDNN with CTC/Attention trained on CommonVoice Italian](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-it)
* [CRDNN with CTC/Attention trained on LibriSpeech](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech)
* [wav2vec 2.0 with CTC trained on Aishell](https://huggingface.co/speechbrain/asr-wav2vec2-ctc-aishell)
* [wav2vec 2.0 with CTC trained on CommonVoice German](https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-de)
* [Transformer trained on Aishell](https://huggingface.co/speechbrain/asr-transformer-aishell)
* [Transformer trained on LibriSpeech](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)
  
#### 4. Adversarial attacks
To generate Adversarial Examples, we utilized [RobustSpeech](https://github.com/RaphaelOlivier/robust_speech), a repository that contains a PyTorch implementation of all considered attacks in our paper.
Please refer to their website for instructions on how to generate adversarial examples. 

## Computing Characteristics

## Building binary Classifiers
