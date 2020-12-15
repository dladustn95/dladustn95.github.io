---
title:  "BART 논문 리뷰"
excerpt: "BART 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, Transformer, NLP]
date: 2020-04-14T18:00:00+09:00
last_modified_at: 2020-04-14T18:00:00+09:00
---

# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

## 1. Introduction
랜덤한 단어가 mask되어 있는 문장을 다시 복원하는 Masked language model과 denoising auto-encoder가 좋은 성능을 보인다.
하지만 특정 task에 집중하여 적용할 수 있는 분야가 한정되는 단점이 있다.  

BART는 넓은 분야에 적용할 수 있도록 seq2seq 구조로 만들어진 denoising auto-encoder다. 
Pretrain은 noise function으로 손상된 text를 복구하도록 모델을 학습시키는 방법으로 이뤄진다.  

![그림1](/assets/images/bart_figure1.png "그림1"){: .align-center}  
BERT는 bidirection encoder로 noise된 token을 예측하는데 generation task에서 사용이 어렵다. 
GPT는 autoregressive하게 다음 token을  예측해 generation에 사용이 가능하지만 bidirectional 정보를 얻지 못한다.
BART는 손상된 text를 입력받아 bidirectional 모델로 인코딩하고, 정답 text에 대한 likelihood를 autoregressive 디코더로 계산한다.  
이런 설정은 noising이 자유롭다는 장점이 있다. 
이 논문에서는 문장 순서를 바꾸거나 임의 길이의 토큰을 하나의 mask로 바꾸는 등의 여러 noising 기법을 평가한다.

## 2. Model
### 2.1 Architecture
BART는 seq2seq 트랜스포머 구조를 사용했고, ReLU activation function을 GeLUs로 변경했다. 
base model은 6 layer, large model은 12 layer를 사용했다. 
디코더의 각 레이어에서는 인코더의 마지막 hidden layer와 cross-attention을 한다. (기존의 트랜스포머 디코더와 동일함)
BERT는 word prediction을 위해 추가로 feed-forward 레이어를 추가했는데 BART는 그렇지 않다.

### 2.2 Pretraining BART
BART는 손상된 text로 학습하는데 디코더의 출력과 원본 text의 loss를 줄이도록 한다. 
다른 Auto-Encoder 모델과 다르게 모든 종류의 noise를 적용할 수 있다.  

![그림2](/assets/images/bart_figure2.png "그림2"){: .align-center}  
이 논문에서는 그림과 같이 5가지의 noise 기법을 사용했다. 
- Token Masking: BERT처럼 랜덤 토큰을 masking하고 이를 복구하는 방식이다.
- Token Deletion: 랜덤 토큰을 삭제하고 이를 복구하는 방식이다. 
Masking과의 차이점은 어떤 위치의 토큰이 지워졌는지 알 수 없다는 점이다.
- Text Infilling: 포아송 분포를 따르는 길이의 text span을 생성해서 이를 하나의 mask 토큰으로 masking 한다. 
즉 여러 토큰이 하나의 mask 토큰으로 바뀔 수 있고 길이가 0인 경우에는 mask 토큰만 추가될 수도 있다. 
SpanBERT에서 아이디어를 얻었는데 SpanBERT는 span의 길이를 알려주었으나 여기서는 알려주지 않고 모델이 얼마나 많은 토큰이 빠졌는지 예측하게 한다.
- Sentence Permutaion: Document를 문장 단위로 나눠서 섞는 방법이다.
- Document Rotation: 토큰 하나를 정해서 문장이 그 토큰부터 시작하게 한다. 
모델이 document의 시작을 구분하게 한다.  

## 3. Fine-tuning BART 
### 3.1 Sequence Classification Tasks
![그림3](/assets/images/bart_figure3.png "그림3"){: .align-center}  
같은 input이 encoder와 decoder에 주어진다. 
디코더의 final hidden state가 새로운 linear classifier로 전달된다. 
이 방법은 BERT의 CLS토큰과 비슷한데 마지막 토큰까지 입력해주면서 전체 입력에 대한 디코더의 attention을 계산할 수 있게했다.  

### 3.2 Token Classification Tasks
전체 document를 인코더와 디코더에 입력한다. 
디코더의 top hidden state를 각 단어에 대한 representation으로 사용한다. 
이 representation을 token classification에 사용한다.  

### 3.3 Sequence Generation Tasks
BART는 autoregressive 디코더를 갖고 있으므로 바로 fine-tuning이 가능하다.
인코더에 input이 주어지면 디코더에서 output을 autoregressive하게 만든다.

### 3.4 Machine Translation
전체 BART 모델을 기계 번역을 위한 pre-trained 디코더로 사용하고 새로운 인코더를 추가해서 인코더-디코더를 fine-tuning 한다. 
새로운 인코더는 외국어를 BART가 학습한 언어로(영어로 학습했으면 영어로) denoising 할 수 있는 입력으로 mapping 한다.
새로운 인코더는 BART와 다른 vocabulary를 사용할 수 있다.
새로운 인코더를 두단계로 학습하는데 두 방법 모두 cross-entropy loss로 backpropagate 한다. 
처음에는 대부분의 BART 파라미터는 그대로 두고 인코더와 BART의 position embedding, BART 인코더의 첫번째 레이어 self-attention input projection matrix만 학습시킨다. 
두번째 단계에서는 모든 파라미터를 학습시킨다.

## 4. Related Work
**논문에서는 related work가 뒤에 나오지만 다음 내용에서 language model과 pre-training 방법을 소개하므로 이해를 돕기 위해 여기서 related work를 다루겠습니다.**  
  
- GPT는 leftward context만 다루기 때문에 몇몇 task에서는 문제가 생긴다. 
- ELMo는 left-only와 right-only representation을 concatenate하는데 두 표현 사이의 상관관계는 학습하지 않는다.
- GPT2는 아주 큰 language model이 unsupervised, multitask 모델처럼 동작하는 것을 보였다.
- BERT는 좌우 context word의 상관관계를 학습하는 masked language modelling을 소개했다. 
학습을 오래하거나(RoBERTa), 레이어의 파라미터를 공유하는 방법(ALBERT), 단어를 masking 하는 대신 공간을 masking 하는 방법(SpanBERT)이 더 향상된 성능을 보였다. 
BERT는 예측이 auto-regressive 하지 않아서 생성 task에는 약하다.

![그림3](/assets/images/bart_figure4.png "그림4"){: .align-center}  
- UniLM은 unidirectional LM, Bidirect LM, sequence LM을 앙상블한 모델이다. 
각 LM task 사이의 파라미터와 모델 구조를 통일함으로써, 여러 LM을 만들어야 했던 필요성을 완화합니다.
BART처럼 생성과 분류 task 모두 가능하다. 
BART와 차이점은 UniLM의 prediction은 conditionally independent하다는 점이다. 
BART는 항상 완전한 입력이 디코더에 주어져서 pre-training과 생성의 차이가 적다. 

![그림3](/assets/images/bart_figure5.png "그림5"){: .align-center} 
- MASS는 BART와 가장 유사한 모델이다. 
연속된 span이 masking된 문장을 인코더 입력으로 주고, 디코더에서 masking 되었던 토큰들을 예측한다.

- XL-Net은 mask된 토큰을 섞인 순서로 auto-regressive하게 예측하도록 BERT를 확장했다. 

## 5. Comparing Pre-training Objectives
BART 모델에는 여러 noising 기법을 사용할 수 있다. 
따라서 이 기법들을 비교해본다.  

### 5.1 Comparison Objectives
- Language Model: GPT와 비슷하다. left-to-right 트랜스포머 모델을 학습시킨다. 
이 모델은 cross-attention이 빠진 BART 디코더와 같다.
- Permuted Language Model: XL-Net을 기반으로 한다. 
1/6 토큰을 샘플링하고 이것을 랜덤한 순서로 auto-regressive하게 생성한다.
- Masked Language Model: BERT처럼 15% 토큰을 mask 토큰으로 바꾸고 독립적으로 이 토큰을 예측하게 한다.
- Multitask Masked Language Model: UniLM처럼 self-attention mask를 추가해서 masked language model을 학습한다.
self-attention mask는 1/6은 left-to-right, 1/6은 rignt-to-left, 1/3은 unmasked로 적용되고,
나머지 1/3은 처음 50% 토큰에는 mask가 없고 나머지 토큰에는 left-to-right mask를 적용한다.
- Masked Seq-to-Seq: MASS와 비슷하다. 토큰의 50%를 포함하는 span에 mask를 하고 mask된 토큰을 예측하는 seq-to-seq 모델을 학습시킨다.
  
일반적인 seq-to-seq task처럼 source를 인코더에 주고 target을 디코더 output으로 하는 방법과 source를 디코더 target의 prefix로 주고 target 부분만 loss를 계산하는 방법으로 학습시켰다.
전자의 방법이 BART 모델에 더 잘했고 후자는 나머지 모델에 더 잘했다.  

### 5.2 Tasks
- SQuAD: Extractive QA task. 주어진 document에서 정답을 추출한다. 
BERT와 유사하게 질문과 document를 concatenate해서 BART 인코더, 디코더 입력으로 준다. 
Classifier를 포함하는 모델이 정답의 시작과 끝 토큰 인덱스를 예측한다.
- MNLI: Bitext classification task다. 두 문장의 의미적 관계를 분류하는 task. 
두 문장을 concatenate하고, eos 토큰을 추가해서 BART 인코더 디코더에 입력한다.
eos 토큰의 representation이 문장의 관계를 예측하는데 사용된다. 
- ELI5: Abstractive QA task. 질문과 document를 사용해 정답을 생성한다. 
- Xsum: Abstractive summary task. 
- ConvAI2: Persona를 사용하는 대화 생성 task.
- CNN/DM: 뉴스 요약 task.  

### 5.3 Results
![그림3](/assets/images/bart_figure6.png "그림6"){: .align-center}  
Pre-trained 모델의 성능은 task가 큰 영향을 미친다. 
예를 들어 language model의 경우 generation task인 ELI5에서는 가장 좋았으나 classification task인 SQuAD에서는 가장 나빴다.  

Token masking이 중요하다.
Document rotation이나 sentence shuffling 기법만 사용했을때 성능은 안좋았다. 
Token deletion이나 masking 방법을 사용한 것이 더 좋은 성능을 보였고 generation task에서는 deletion이 masking 보다 더 좋은 성능을 보였다.  
Left-to-right pre-training이 generation 성능을 향상시킨다.
Masked Language Model과 Permuted Language Model은 다른 모델에 비해 generation 성능이 떨어진다.
그 이유를 논문에서는 left-to-right auto-regressive 모델링이 포함되지 않았기 때문이라고 주장한다.  
SQuAD에는 Bidirectional 인코더가 중요하다.
이후 문맥의 정보(future context)가 중요하기 때문에 이 정보를 얻지 못하는 left-to-right 디코더는 성능이 좋지 않다.  
Pre-training의 방법만이 중요한 것이 아니다. 
논문에서 구현한 Permuted Language Model은 XL-Net보다 성능이 떨어지는데 그 이유는 XL-Net에 적용된 여러 기법(relative-position embedding 등)이 사용되지 않기 때문이다.  
Language Model이 ELI5에서 가장 좋은 성능을 보인다.
ELI5는 다른 generation task에 비해 높은 PPL을 가지고, 다른 모델이 BART보다 뛰어난 성능을 보인다. 
그 이유를 논문에서는 output이 input과 적은 연관성을 가지기 때문이라고 주장한다.  
BART가 ELI5를 제외한 모든 task에서 가장 좋은 성능을 가진다.

## 6. Large-scale Pre-training Experiments
최근 연구에서 큰 batch size와 corpora를 사용해 pre-training하는 것이 성능의 향상을 이끌어낸다고 한다.
이를 실험하기 위해 BART를 RoBERTa 모델과 같은 규모로 실험했다.

### 6.1 Experimental Setup
Large 모델을 12레이어, 1024 hidden state를 갖도록 설정한다.
RoBERTa처럼 batch size로 8000을 설정했고, 모델을 50만번 학습했다.
Document는 GPT2와 같은 byte-pair encoding을 사용해 토크나이징했다. 
위의 실험 결과를 참고해 Text infilling과 sentence shuffling을 섞어서 pre-training 했다.
각 document의 30% 토큰을 masking 했고, 모든 문장을 섞었다. 
마지막 10%의 training step에서는 dropout을 적용하지 않았다.
RoBERTa와 같은 데이터를 사용했다.  

### 6.2 Discriminative Tasks
![그림3](/assets/images/bart_figure7.png "그림7"){: .align-center}  
SQuAD, GLUE에 대해 BART 성능을 측정했다. 
같은 데이터를 사용해 다른 방법으로 학습한 RoBERTa와 비교해보면 큰 차이가 없음을 보였다.
BART는 generation task에서 성능 향상을 이루면서 classification task에서도 경쟁력을 유지했다.  

### 6.3 Generation Tasks
![그림3](/assets/images/bart_figure8.png "그림8"){: .align-center}  
CNN/DM은 주로 앞의 3문장이 정답으로 사용되는 extractive한 데이터이다. 
XSum은 abstractive한 데이터이다. 
두 데이터에서 BART가 가장 좋은 성능을 보였다. 특히 Xsum에서 크게 향상되었고, 정답의 질도 향상되었다.  

![그림3](/assets/images/bart_figure9.png "그림9"){: .align-center}  
Persona를 반영하는 데이터셋인 ConvAI2로 실험했다.
BART가 다른 모델보다 더 잘했다.  

![그림3](/assets/images/bart_figure10.png "그림10"){: .align-center}  
Abstractive QA task에서는 ELI5 데이터셋을 사용했다.
BART가 더 잘하긴 했지만 정답과 질문의 연관성이 약하기 때문에 더 연구해볼 필요가 있다.

### 6.4 Translation
![그림3](/assets/images/bart_figure11.png "그림11"){: .align-center}  
WMT16 RomanianEnglish 데이터셋을 사용했다.
앞서 언급한 방법으로 fine-tuning을 진행했다.
back-translation 데이터가 없으면 덜 효과적이었고, overfitting 되는 경향이 있었다.
따라서 추가적인 연구가 필요하다. 

## 7. Qualitative Analysis
![그림3](/assets/images/bart_figure12.png "그림12"){: .align-center}  
BART는 특히 summarization task에서 성능이 많이 향상됐다.
위의 예시는 BART로 생성한 요약의 예이다. 
Background knowledge를 사용해 요약을 더 잘하는데 특히 source에도 없는 지식을 사용할 수 있다.

## 8. Conclusions
BART는 분류 task에서 RoBERTa와 비슷한 성능을 내면서도 generation task에서도 state-of-the-art 성능을 보였다.
향후 연구로 pre-training을 위한 document를 손상시키는 방법을 더 조사해야할 필요가 있다.  