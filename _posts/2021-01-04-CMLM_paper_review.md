---
title:  "Mask-Predict: Parallel Decoding of Conditional Masked Language Models 논문 읽기"
excerpt: "Mask-Predict: Parallel Decoding of Conditional Masked Language Models 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, MachineTranslation, Transformer, NLP]
date: 2021-01-04T18:00:00+09:00
last_modified_at: 2021-01-04T18:00:00+09:00
---

# Mask-Predict: Parallel Decoding of Conditional Masked Language Models

## 1. Introduction
이 논문은 Conditional masked language model을 제안한다. 
일정한 수만큼 디코딩 과정을 반복하며 mask된 토큰을 예측하는 방법으로 parallel 디코딩을 하는 알고리즘을 제안한다. 
기존의 parallel 디코딩은 전체 문장을 한 번에 생성하지만 mask-predict는 생성한 문장에 masking을 하고 다시 예측하는 것을 반복한다. 
완전히 mask된 문장에 들어갈 토큰을 parallel하게 예측하는 것부터 시작해서 일정 수 만큼 반복하고 끝난다. 
이 방법은 모델이 양방향 문맥 정보를 활용해 토큰을 예측할 수 있게 한다.

## 2. Conditional Masked Language Models
source 문장 $$X$$와 target 문장의 주어진 토큰 $$Y_{obs}$$를 사용해 target 문장의 mask 토큰$$Y_{mask}$$에 들어갈 단어를 예측한다.  
$$P(y|X,Y_{obs})$$  
여기서 각 mask 토큰은 각각 독립이라고 가정한다.
모델은 또한 target 문장의 길이에 conditioning 된다. 
$$N=|Y_{mask}|+|Y_{obs}|$$

### 2.1 Architecture
일반적인 Encoder-Decoder 형태의 트랜스포머를 사용한다. 
디코더에서는 일반적인 AR 모델의 트랜스포머 디코더와 다르게 어텐션 마스크를 제거한다. 
NAR 방식으로 디코딩을 하기 때문에 left-to-right가 아닌 bi-directional한 디코더를 사용할 수 있다.

### 2.2 Training Objective
학습 과정에서는 target 토큰에서 랜덤하게 마스크할 토큰 $$Y_{mask}$$를 고른다.
먼저 1~sequence length에서 uniform distribution으로 숫자를 골라 마스크할 토큰의 수를 정한다. 
그 수만큼 랜덤한 토큰을 골라 mask 토큰으로 바꾼다. 
Mask 토큰에 대해서만 cross-entropy loss를 계산한다. 
$$L_{token}=-\sum_{i=1}^{|Y_{mask}|}\log P(y_i|X,Y_{obs})$$

### 2.3 Predicting Target Sequence Length
기존의 AR 모델의 경우 끝을 뜻하는 EOS 토큰이 나올 때까지 디코딩 과정을 진행하기 때문에 따로 길이를 구할 필요가 없었다. 
그러나 제한하는 CMLM과 같이 NAR 모델은 전체 문장을 parallel하게 디코딩하므로 길이를 사전에 알아야 한다. 
이 논문에서는 인코더에 길이를 구하기 위한 special LENGTH 토큰을 삽입한다. 
이 special 토큰은 BERT의 CLS 토큰과 비슷하게 인코더의 전체 문장 정보를 담는 역할을 한다. 
LEGNTH 토큰의 hidden state를 linear layer에 넣어 디코딩할 문장의 길이를 구한다.  
길이에 대한 loss를 앞서 구한 target 토큰에 대한 loss에 더해준다. 

## 3. Decoding with Mask-Predict
Mask-predict 디코딩 알고리즘은 전체 문장을 일정한 수만큼 반복하여 디코딩한다. 
각 iteration마다 mask할 토큰을 선택하고 그 토큰을 다시 예측한다. 
Mask 되는 토큰은 confidence score가 가장 낮은 토큰들로 모델이 더 많은 정보를 가지고 다시 예측하게 한다. 

### 3.1 Formal Description
The target sequence’s length $$N$$  
The target sequence $$(y_1,…,y_N)$$  
The probability of each token $$(p_1,…,p_N)$$  
Number of iterations $$T$$  
  
**Mask**  
처음에는$$(t=0)$$ 모든 토큰을 mask한다. 
그 후로는 가장 낮은 probability score를 갖는 토큰 $$n$$개를 mask한다.  
$$Y_{mask}^{(t)}=\arg\min_i (p_i,n)$$  
$$Y_{obs}^{(t)}=Y\backslash_{mask}^{(t)}$$  
mask할 토큰의 수 $$n$$은 linear하게 구한다.  
$$n=N\cdot\frac{T-t}{T}$$, $$T$$는 전체 iteration의 수이다.  
  
**Predict**  
mask를 한 뒤 CMLM을 사용해 mask된 토큰 $$Y_{mask}^{(t)}$$을 예측한다. 
mask된 각 토큰 $$y \in Y_{mask}^{(t)}$$에 대해 가장 높은 확률을 갖는 값을 선택하고 그 probability score를 업데이트 한다.  
$$Y_{i}^{(t)}=\arg\max_w P(y_i=w|X,Y_{obs}^{(t)})$$  
$$p_{i}^{(t)}=\max_w P(y_i=w|X,Y_{obs}^{(t)})$$  
mask되지 않은 토큰은 바뀌지 않는다. 

### 3.2 Example
![그림1](/assets/images/CMLM_figure1.png "그림1"){: .align-center}  
첫 iteration에서는 전체 target sequence가 mask 되어 있기 때문에 pure Non-Autoregressive하게 문장을 생성했다.  
$$P(Y_{mask}^{(0)}|X,Y_{obs}^{(0)})=P(Y|X)$$  
이 단계에서는 반복을 포함해서 문법적으로 맞지 않는 부분이 있다. 
이는 기존의 Non-Autoregressive 디코딩 방법의 multi-modality 문제이다.  
다음부터는 unmasked된 토큰에 condition 되어 mask 토큰을 예측한다. 
그래서 iteration이 진행될수록 보다 자연스러운 번역이 가능해진다. 

### 3.3 Deciding Target Sequence Length
앞서 설명한 것처럼 target sequence의 길이를 구하기 위해 LENGTH 토큰을 사용했다. 
하지만 정확히 예측하지 못하기 때문에 여러개의 length candidate를 사용했다. 
각각의 candidate에서 생성한 문장 중 가장 높은 average log-probability를 가진 문장을 사용한다.  
$$\frac{1}{N} \sum\log p_i^{(T)}$$  

## 4. Experiments
### 4.1 Translation Benchmarks
![그림2](/assets/images/CMLM_figure2.png "그림2"){: .align-center}  
![그림3](/assets/images/CMLM_figure3.png "그림3"){: .align-center}  
좋은 성능을 보였다. 

### 4.2 Decoding Speed
![그림4](/assets/images/CMLM_figure4.png "그림4"){: .align-center}  
다양한 iteration과 length candidate를 설정해 속도를 비교해보았다. 
흔히 말하는 decoding speed와 translation quality 사이의 trade-off 관계가 있다는 것을 볼 수 있었다. 
Autoregressive 모델은 fairseq에서 state를 캐싱해 디코딩 속도를 높이는 방법을 사용했음에도 유의미한 속도 향상이 있었다. 

## 5. Analysis
### 5.1 Why Are Multiple Iterations Necessary?
Multi-modality 문제는 입력에 대해 여러가지 번역이 가능할 때 생기는 문제이다. 
간단한 예를 들면 "안녕"을 영어로 번역할 때 "Hi"와 "Hello"로 번역 가능한 것이다. 
이 문제는 같은 단어가 반복되는 등의 형태로 나타난다. 
기존의 Non-Autoregressive 방법들은 각 토큰을 conditional independent하게 보았기 때문에 이 문제로 어려움을 겪었다. 
Mask-predict 방법을 사용하는 경우 output 토큰의 일부에 condition된 문장을 생성할 수 있기 때문에 이 문제를 완화시켜준다.  
![그림5](/assets/images/CMLM_figure5.png "그림5"){: .align-center}  
그림 5에서 볼 수 있듯이 iteration이 진행될수록 토큰이 반복되는 현상이 적어지고 번역의 질 또한 향상된다. 

### 5.2 Do Longer Sequences Need More Iterations?  
![그림6](/assets/images/CMLM_figure6.png "그림6"){: .align-center}  
문장이 길수록 더 많은 iteration이 필요한지 조사했다. 
Iteration의 수 T가 커질수록 성능이 향상되는 것을 볼 수 있었지만 4번의 iteration만으로도 충분히 좋은 결과를 얻을 수 있었다. 

### 5.3 Do More Length Candidates Help?  
![그림7](/assets/images/CMLM_figure7.png "그림7"){: .align-center}  
Target sequence의 길이를 구하기 위해 length candidate 방법을 사용했다. 
더 많은 length candidate를 둘수록 성능이 향상되는지 분석했다. 
그림7에서 length candidate가 커지면 성능이 향상되다가 다시 줄어드는 것을 볼 수 있다. 
이는 길이가 너무 짧은 문장이 높은 average log-probability를 가지면서 성능이 줄어드는 것으로 분석됐다. 
또한 가장 좋은 성능을 가지는 length candidate=3 일때도 문장의 정확한 길이를 예측하는 확률은 약 40%밖에 되지 않았다. 

### 5.4  Is Model Distillation Necessary?
![그림8](/assets/images/CMLM_figure8.png "그림8"){: .align-center} 
Dataset의 노이즈 때문에 기존의 AR 모델이 생성한 결과를 사용해 학습하는 distilation 방법이 제안되어 왔다. 
각각의 데이터에 대해 성능을 측정한 결과 distillation data를 사용한 것이 더 좋은 성능을 보였다. 

## 6. Conclusion
새로운 방법인 conditional masked language model과 mask-predict 디코딩 알고리즘을 제안했다. 
하지만 target sequence의 길이를 구하는 것과 distillation data에 의존적인 부분에 대해 개선이 필요했다. 