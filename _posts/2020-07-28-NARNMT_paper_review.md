---
title:  "Non-Autoregressive Neural Machine Translation 논문 리뷰"
excerpt: "Non-Autoregressive Neural Machine Translation 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, MachineTranslation, Transformer, NLP]
date: 2020-07-28T18:00:00+09:00
last_modified_at: 2020-07-28T18:00:00+09:00
---

# Non-Autoregressive Neural Machine Translation

## 1. Introduction
&nbsp;&nbsp;신경망 모델은 기계 번역에서 기존 통계 기반 모델보다 더 우수하다. 
그러나 inference time이 길다는 단점이 있다. 
한 번에 한 단계씩 생성하는 Autoregressive(AR) 디코더 사용하기 때문이다. 
AR 디코더는 각 토큰을 생성할 때 이전에 생성한 토큰에 영향을 받는다. 
따라서 병렬화(parallelizable)가 불가능하다.  
기존 Transformer 인코더에 fertilities, sequences of number를 예측하는 모듈 추가한 새로운 None-Autoregressive(Non-AR) 모델 제안. 
Fertilities는 지도학습되고, inference time 동안 전체적으로 일관된 plan과 함께 디코더에 주어져서 동시에 output을 계산 가능하게 함.

## 2. Background
### 2.1 Autoregressive Neural Machine Translation
입력 $$X={x_1, ..., x_{T'}}$$가 주어지면 생성가능한 $$Y={y_1, ..., y_{T'}}$$에 대한 probability를 chain of condition probability를 통해 구한다.  
$$p_{\mathcal{AR}}(Y|X;\theta)=\prod_{t=1}^{T+1}p(y_t|y_{0:t-1},x_{1:T'};\theta)$$  

스페셜 토큰 \<bos>, \<eos>가 문장의 시작과 끝을 알리기 위해 사용된다. 
이런 확률은 neural net으로 표현됨.  
**Maximum Likelihood training**  
각 decoding step마다 Maximum Likelihood를 적용
$$\mathcal{L}_{ML} = \log p_{\mathcal{AR}}(Y|X;\theta)=\sum_{t=1}^{T+1}p(y_t|y_{0:t-1},x_{1:T'};\theta)$$  

**Autoregressive NMT without RNNs**  
훈련 과정에서는 전체 target을 알고 있기 대문에 조건부확률을 계산하기 위해 이전 output word에 의존할 필요는 없다. 
그래서 훈련 중에는 병렬 처리 가능.

### 2.2 Non-Autoregressive Decoding
**Pros and cons of autoregressive decoding**  
기존 모델에서 사용하는 방식은 인간이 언어를  생성하는 방식과 유사하다. 
큰 corpora에서 학습시키기 쉽고 beam search로 적절한 output을 찾을 수 있다. 
그러나 순차적으로 실행되어야 해서 오래 걸린다. 
Beamsearch는 beam size와 관련한 문제가 있고 beam간의 의존성이 존재해 병렬화가 제한된다.  

**Towards non-autoregressive decoding**  
![그림1](/assets/images/NARNMT_figure1.png "그림1"){: .align-center}  
나이브한 방법은 모델에 존재하는 AR connection을 제거하는 것이다. 
Target sequence 길이 T는 분리된 조건부확률로 표현할 수 있다고 가정하자. 
$$p_{\mathcal{NA}}(Y|X;\theta)=p_L(T|x_{1:T'};\theta)\cdot\prod_{t=1}^{T}p(y_t|y_{0:t-1},x_{1:T'};\theta)$$  
이 식은 likelihood로 표현할 수 있고 CELoss로 각 output distribution마다 독립적으로 훈련할 수 있다. 
그리고 inference time에 병렬적으로 계산 가능하다

### 2.3 The Multimodality Problem
위의 나이브한 방법은 완전히 조건부 독립이기 때문에 좋은 결과를 얻지 못한다. 
각 토큰 distribution은 source 문장 X에만 의존하기 때문이다. 
즉 사람들에게 문장을 주고 각자 한 단어씩만 번역해 문장을 완성하는 것과 비슷하다.  
또한 한 단어가 여러 단어로 번역이 가능한 경우, 이것에 대한 target distribution은 여러 단어에 대한 독립확률분포로 나타낼 수 없다. 
조건부독립분포는 다른 단어를 허용하지 않기 때문이다. 
이를 multimodal problem으로 정의하고 설명할 것이다.

## 3. The Non-Autoregressive Transformer (NAT)
![그림2](/assets/images/NARNMT_figure2.png "그림2"){: .align-center}  
Non-Autoregressive Transformer 제안. 
4개의 모듈로 이뤄짐 encoder, decoder stack, 새롭게 추가되는 fertility predictor, 토큰 디코딩을 위한 a translation predictor

### 3.1 Encoder Stack
각 encoder, decoder stack은 트랜스포머와 유사. Encoder는 트랜스포머와 똑같다.

### 3.2 Decoder Stack
**Decoder Inputs**  
decoding하기 전에 출력 문장의 sentence 길이를 알아야 한다. 
중요한건 time-shifted target output이나 prevoius predicted output을 decoder layer의 입력으로 사용 못한다는 점이다. 
decoder layer에 입력을 생략하거나 position embedding만 사용하면 성능이 좋지 않다. 
대신 encoder에서 쓰인 source input을 사용한다. 
Source와 target의 길이가 다르므로 두가지 방법을 사용한다.
- Copy source inputs uniformly : 일정한 속도로 input을 복사함 
- Copy source inputs using fertilities : input을 0~여러 번 복사하는 것. 각 input이 복사되는 횟수는 그 단어의 “fertility”라고 할 수 있다. 
 Result 길이는 fertility의 합으로 정해진다. 

**Non-causal self-attention**  
decoding step에서 이후 단계 정보에 접근하는 것을 막을 필요가 없다. 
대신 각 쿼리 위치에서 자기 자신을 attention 하는 것을 mask한다. 
Unmasked self-attention에 비해 디코더 성능을 향상 시킬 수 있다. 

**Positional attention**  
decoder layer에 positional attention module을 추가함.  
$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_{model}}})\cdot V$$   
Positional encoding은 q,k로 decoder state는 v로 사용. 
위치 정보를 attention process에 통합하고 embedding layer에만 주는 것보다 더 강한 정보를 준다. 
이게 decoder의 local reordering 능력을 증가시킨다. 

### 3.3 Modeling Fertility to Tackle the Multimodality Problem
Latent variable z를 사용해 multimodality 문제를 해결할 수 있다. 
Z는 특정 input-output이 주어지면 쉽게 구할 수 있어야 함. 
Conditioning context에 z를 추가하면 marginal probability가 최대한 근접해야 함. 
Output translation을 직접적으로 설명하면 안된다.
Fertility를 사용해 z를 구현. 
Fertility를 이용해 input을 복사할 때 자연스럽게 z를 도입함.  
$$p_{\mathcal{NA}}(Y|X;\theta)=\sum_{f1,...,f_{T'}\in\mathcal{F}}\left(\prod_{t'=1}^{T'}p_F(f_{t'}|x_{1:T'};\theta)\cdot \prod_{t'=1}^{T'}p_F(y_t|x_1\{f_1\},...,x_{T'}\{f_{T'}\};\theta)\right)$$  
Fertility prediction : 마지막 encoder layer에 하나의 신경망을 사용해서 구했다. Fertility는 각 단어의 속성이지만 전체 문장 정보와 context에 의존해서 모델링함.
Benefits of fertility : 앞서 말한 3가지 속성을 모두 만족한다. 
Fertility를 latent variable로 사용한다는 것은 translation의 길이를 분리해서 고려할 필요가 없다는 것을 의미한다. Fertility 크기를 sampling하는 것으로 모델이 다양한 translation을 가능하게 함.

### 3.4 Translation Predictor and the Decoding Process
inference 과정에서 모든 latent fertility sequence에 대해 marginalizing해서 가장 높은 조건부 확률을 가지는 translation을 식별한다. 
Fertility 시퀀스가 주어지면 각 output 위치에 대한 local probability만 독립적으로 최대화하면 최적 translation을 찾을 수 있다. 
전체 공간을 탐색하는 것은 한계가 있기 때문에 3가지 방법을 제안한다. 
**Argmax decoding**  
각 입력 단어에 대한 가장 높은 확률의 fertility 선택
**Average decoding**  
softmax 확률의 평균으로 추정.
**Noisy parallel decoding (NPD)**  
fertility space에서 noise를 주입한 여러 샘플을 만들고 가장 높은 점수를 갖는 sample 고른다.
AR teacher로 최상의 translation을 구별한다. 
NPD는 stochastic한 검색 방법이고 sample size만큼 계산량을 증가시킨다. 

## 4. Training
![그림3](/assets/images/NARNMT_figure3.png "그림3"){: .align-center}  
translation model과 fertility neural network model 2가지로 나눌 수 있다.
Translation model과 fertility model을 동시에 학습한다.
q defined by a separate, fixed fertility model. 
Training corpus의 pair(X,Y)에 대한 sequence를 생성하는 External aligner의 output이나 teacher model의 attention weight에서 계산된 fertility를 사용 가능하다. 
q가 결정되어있어서 추론(학습)과정을 간소화할 수 있다.

### 4.1 Sequence-Level Knowledge Disstillation
latent fertility model이 multimodal problem을 해결하는데 도움은 되지만 완벽한 해결방법은 아니다. 
Single sequence of fertility에 여러 정답이 존재할 수 있다. 
이를 해결하기 위해 AR teacher 모델의 greedy output으로 만든 새로운 corpus에 대해 모델을 학습시킨다. 
따라서 항상 같은 결과로 학습할 수 있지만 기존 dataset보다는 퀄리티가 떨어진다.

### 4.2 Fine Tuning
![그림4](/assets/images/NARNMT_figure4.png "그림4"){: .align-center}  
Deterministic한 external alignment system에 의존하는 경향이 있기 때문에 전체 모델을 학습시키기 위해 fine tuning을 진행한다.
Word-level distillation 형태를 띄는 새로운 loss term 제안한다.  
Highly peaked student에 더 적합한 reverse KL을 사용해 translation model fine-tuning한다.
일반적인 Distillation loss와 두가지 텀을 weighted sum. 
하나는 base line으로 정규화된 예측한 fertility distribution, 하나는 external fertility inference model.

## 5. Experiments
### 5.1 Experimental Settings
IWSLT16 En–De, WMT14 En–De, WMT16 En–Ro을 사용했다. 
Ablation study는 가장 작은 IWSLT로 했다. 
WMT에 대해서는 양방향으로 학습 및 실험했다.
Teacher model로 AR 모델 (SOTA transformer) 사용했고, Student는 같은 크기와 hyperparameter를 가진다. 
Teacher와 student 모델이 같은 인코더를 사용한다. 기존 transformer 모델 하이퍼 파라미터를 가져옴. 
Fertility prediction을 supervise하기 위해 IBM2에서 구현된 fast_align을 사용했다.

### 5.2 Results
![그림5](/assets/images/NARNMT_figure5.png "그림5"){: .align-center}  
bleu 점수가 2~5점 밖에 차이 안나고 속도는 greedy보다는 10배 beam보다는 15배 더 빠르다.

### 5.3 Ablation study
![그림6](/assets/images/NARNMT_figure6.png "그림6"){: .align-center}  
Decoder input으로 posAtt만 넣으면 학습이 안된다.
Distillation corpus가 ground truth를 사용하는 것보다 성능이 더 좋다.
디코더 입력으로 uniform보다 fertility가 더 좋다.
finetuning과정에서 RL이나 BP 하나만 사용하면 수렴이 안된다. 
복잡한 문장에서 주로 단어 반복이 나오는 경우가 보인다.
NPD에서 반복이 안보이는 걸 보면 teacher 모델을 이용한 scoring이 오류를 어느정도 없애주는 것을 알 수 있다.