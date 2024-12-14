---
title:  "Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance? 논문 읽기"
excerpt: "Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance? 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, EMNLP23]
date: 2024-01-25T18:00:00+09:00
last_modified_at: 2024-01-25T18:00:00+09:00
---
  
# Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance?

## Abstract

PLM(사전 훈련된 언어 모델)이 언어에 대해 학습하는 방법과 내용을 이해하는 것은 자연어 처리 분야에서 열려 있는 과제입니다. 이전 작업에서는 의미론적 및 구문론적 정보를 캡처하는지 여부와 데이터 또는 사전 학습 목표가 성능에 어떤 영향을 미치는지 식별하는 데 중점을 두었습니다. 그러나 우리가 아는 한, 입력 토큰 문자의 정보 손실이 PLM 성능에 어떤 영향을 미치는지 구체적으로 조사한 이전 연구는 없습니다. 본 연구에서는 개별 토큰의 작은 문자 하위 집합을 사용하여 언어 모델을 사전 학습함으로써 이러한 격차를 해결합니다. 놀랍게도 우리는 극단적인 설정, 즉 각 토큰의 한 문자만 사용하는 사전 훈련에서도 전체 토큰 모델과 비교하여 표준 NLU 벤치마크 및 probing 작업의 성능 유지율이 높다는 것을 발견했습니다. 예를 들어, 토큰의 첫 번째 문자 하나만으로 사전 학습된 모델은 SuperGLUE 및 GLUE 작업에서 각각 전체 토큰 모델의 약 90% 및 77%의 성능 유지를 달성합니다.

## TL; DR

- 학습 데이터의 단어에서 일부 글자만 사용하여 모델을 학습하였을 때의 성능 및 분석을 진행함
- PLM이 언어를 이해하는 방식이 사람과 유사한 점도 존재하고, 다른 특성을 갖는 점도 있음

## Introduction

![그림1](/assets/images/URIT/1.png "그림1"){: .align-center}  

Pre-trained 모델이 language를 배우거나 표현하는 방법은 완전히 알려지지 않았다. 이 중에 pre-training data의 양과 질이 성능에 미치는 영향이 주로 연구되었다. 

이전 연구들에서는 PLM이  pre-training data의 개별 토큰에서 실제로 필요로 하는 정보량을 조사한 적이 없다. 이 논문에서는 token의 information loss를 주고 이 것이 미치는 영향을 조사한다. Information loss는 토큰을 글자수 단위의 subset으로 표현하는 방법을 사용한다. (토큰의 1~3글자만 사용, 자음, 모음만 사용)

인간이 각 단어의 첫 번째 또는 마지막 문자만 읽어 텍스트를 이해하는 등 부분적인 단어 정보를 처리하는 데 유연성을 가지고 있음을 보여준 인지 과학 및 심리 언어학 연구에서 영감을 얻었다. 논문에서는 부분 단어 정보가 제시될 때 인간의 읽기 전략과 언어 모델 간의 유사성을 조사하여 두 시스템의 공통점과 차이점을 밝힌다.

## Models

### Input

Single character
각 토큰의 한 글자만 사용. First (F), Middle (M), Last (L)으로 첫번째, 중간, 마지막 글자만 사용.

Two characters
각 토큰에서 최대 두 글자만 사용. First and last (FL), First two (FF), Last two (LL).

Three characters
각 토큰에서 최대 세 글자만 사용. First, middle and last (FML), First three (FFF), Last three (LLL).

Vowels (V)
토큰의 모음만 사용. PLM이 모음만으로 의미있는 정보를 얻을 수 있는지 조사.

Consonants (C)
토큰의 자음만 사용.

### Architecture and Pre-training

BERT-BASE 모델을 사용하여 성능을 측정했다. 두가지 MLM task를 실험했다.

Predicting masked character subsets:  
mask된 character subsets을 예측하게 하는 방법. 예를들어 photo, phone이 masked token이라면 FF인 경우 ‘ph’, FL인 경우 각각 ‘po’, ‘pe’를 예측한다.

Predicting the original full token:  
입력은 character subsets을 사용하고, mask 된 단어는 original 토큰을 예측하게 하는 방법. 이 방법으로 학습한 PLM을 GLUE와 SuperGLUE로 성능을 측정했을때 두 가지 방법의 성능차이가 존재하지 않았다.

## Experimental Setup

### Tokenization

![그림2](/assets/images/URIT/2.png "그림2"){: .align-center}  

이 논문에서는 단어를 기준으로 하기 때문에 BPE와 같은 subword tokenization 방법을 쓰지 않고, 공백을 기준으로 token을 나눴다. 숫자는 special token으로 바꿨다.

### Data

BookCorpus, English Wikipedia를 pre-training 데이터로 사용. GLUE, SuperGLUE에 존재하는 각각의 task에 대해 fine-tuning.

## Results

![그림3](/assets/images/URIT/3.png "그림3"){: .align-center}  

GLUE 데이터에 대해 fine-tuning 후 성능 측정 결과. 

글자 수가 늘어나면 성능도 함께 증가한다. 각각의 토큰에서 얻을 수 있는 정보량이 증가함에 따라 모델이 잘 학습되는 것으로 예상된다.

첫 글자가 마지막 글자보다 더 좋은 성능을 가진다. 이는 첫 글자가 더 많은 정보를 갖고 있음을 보인다. 마찬가지로 FL이 FF나 LL보다 더 많은 정보를 갖고 있음을 보여준다. 이는 사람의 독해력에서 첫글자와 마지막 글자를 중간 글자보다 더 중요하게 본다는 연구 결과와 비슷하다.

자음만 사용하는 모델이 가장 좋은 성능을 보이는데 이는 다른 모델에 비해 정보 손실이 적었기 때문이다.

한 글자만 사용하는 경우에도 모델의 성능이 높은데 (전체 토큰을 사용하는 것의 77% 수준), 이는 PLM이 아주 작은 입력 정보만으로도 downstream task를 해결할 수 있다는 것을 보여준다. 

![그림4](/assets/images/URIT/4.png "그림4"){: .align-center}  

GLUE와 비슷한 양상을 보인다. PLM은 각 토큰에서의 정보가 많아질 때 잘 학습한다. 

## Analysis

### Pre-training Loss Curves

![그림5](/assets/images/URIT/5.png "그림5"){: .align-center}  

세가지 모델 모두 안정적인 loss curve를 보이기 때문에 아주 제한된 입력 정보만으로도 학습이 된다는 것을 알 수 있다. 

### Corpus-based Token Information Loss

![그림6](/assets/images/URIT/6.png "그림6"){: .align-center}  

Full Token에서 42%의 토큰이 1~3 사이의 글자 길이를 가진다. One char 토큰에서는 전체의 95% 토큰이 1글자로 작아진다. Two chars에서는 79%가 정보를 잃는다. Three chars에서는 48%의 토큰의 정보를 잃는다. 

![그림7](/assets/images/URIT/7.png "그림7"){: .align-center}  

각 카테고리별로 n-gram의 분포를 조사한 결과이다. Full Token과 Consonants가 가장 많은 수의 조합을 가진다. 

### Probing

![그림8](/assets/images/URIT/8.png "그림8"){: .align-center}  

부분 토큰으로 사전 학습한 모델의 Probing 성능이 Full token 모델 보다 낮을 것이라고 가정했다. 입력으로 PLM의 representation을 사용했는데, 모델의 모든 layer에 대한 성능을 측정하고 그 중 가장 좋은 값을 사용했다. 

FML을 사용한 모델이 3가지 task에서 가장 좋은 성능을 기록했다. 사전 학습 과정에서 정보의 손실이 일어났음에도 이러한 성능을 보였다. 

사전 학습에 많은 글자가 사용될 수록 성능이 향상되는 경향도 볼 수 있었다. 

Probing task에서는 첫 글자 보다 마지막 글자가 더 좋은 성능을 보였다. 이는 PLM이 영어의 언어적 정보를 인코딩할 때 단어 내부의 특정 글자 위치가 중요한 역할을 한다고 볼 수 있다.

마지막으로 하나의 글자로만 학습시킨 모델이라도 어떠한 언어적 정보를 인코딩할 수 있는 능력이 있다는 것을 보여줬다. 특히 시제 task에서 Last char 모델이 84.1% 성능을 보여준다.

## Discussion & Conclusion

- First - Last Char 모델이 좋은 성능을 보이는 것은 단어의 첫 글자와 마지막 글자가 인간의 독해에 중요하며, 단어의 인지적 표현에서 글자의 위치가 모두 같은 비중을 갖지 않는다는 인지과학과 심리언어학 연구와 일치하는 것으로 보인다.
- PLM은 인간에게는 이해하기 어려운 데이터에서도 학습할 수 있는 능력을 가진다. 단어에서 1~3 글자만 사용해도 자연어 이해 작업을 잘 수행하거나 의미론적 및 문법적 정보를 학습하는 것으로 보인다. 이는 PLM이 보이지 않는 또는 제한된 데이터에서도 패턴을 찾아내는 능력을 보여준다. 따라서 PLM이 언어를 처리하는 능력이 인간의 인지와 근본적으로 다를 수 있다고 볼 수 있다.
- 하나의 글자만 사용해도 어느 정도의 성능을 얻을 수 있고, probing task에서는 토큰 내부의 특정 글자의 위치가 모델이 언어적 정보를 얻는데 영향을 준다는 것을 알 수 있었다.

#### Probing task

TreeDepth: 구문 분석 트리의 깊이를 예측하여 문장의 계층 구조에 대한 정보를 유지하는지  
Top-Const: 문장 구문 분석 트리의 상위 구성 요소를 예측  
BShift: 인접한 두 단어가 반전되었는지 여부를 평가  
Tense: 주절의 동사가 현재형인지 과거형인지 예측  
SubjNum: 주절의 주어가 단수인지 복수인지를 예측  
CoordInv: 두 개의 좌표절로 구성된 문장이 반전되었는지 여부를 판별  
각 작업은 훈련용 100,000개 문장, 검증용 10,000개 문장, 테스트용 10,000개 문장으로 구성.  