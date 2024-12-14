---
title:  "Pre-training Intent-Aware Encoders for Zero- and Few-Shot Intent Classification 논문 읽기"
excerpt: "Pre-training Intent-Aware Encoders for Zero- and Few-Shot Intent Classification 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, Zero-ShotClassification, EMNLP23]
date: 2024-03-21T18:00:00+09:00
last_modified_at: 2024-03-21T18:00:00+09:00
---
  
# Pre-training Intent-Aware Encoders for Zero- and Few-Shot Intent Classification

## Abstract

의도 분류 (Intent Classification, IC)는 목적 지향 대화 시스템에서 중요한 역할을 합니다. 그러나 IC 모델은 각 사용자 의도에 대해 충분한 주석이 없는 예제 없이 훈련할 때 일반화가 잘 되지 않는 경우가 많습니다. 저희는 intent psuedo-label과 contrastive learning을 사용하여 IC 작업에 적합한 임베딩을 생성하는 텍스트 인코더를 위한 새로운 사전 훈련 방법을 제안하여 사람이 label을 생성하는 작업이 필요하지 않습니다.이 사전 훈련 전략을 적용함으로써 우리는 Pre-trained Intent-aware Encoder (PIE)라는 새로운 모델을 소개합니다. 이 모델은 발화의 인코딩을 해당 의도 이름과 일치시키도록 설계되었습니다. 구체적으로, 우리는 먼저 태거를 훈련하여 의도를 해석하는 데 중요한 발화 내 핵심 구절을 식별합니다. 그런 다음 이러한 추출된 구절을 대조적인 방식으로 텍스트 인코더를 사전 훈련하는 예제로 사용합니다. 결과적으로 우리의 PIE 모델은 네 개의 IC 데이터셋에 대해 N-way zero- 및 one-shot 설정에서 이전 사전 훈련 텍스트 인코더보다 최대 5.4% 및 4.0% 더 높은 정확도를 달성합니다.

## TL; DR

- Utterance에서부터 pseudo intent name을 생성하는 알고리즘과, 이와 연관된 데이터셋을 공유함.
- Gold and pseudo intent names에 Intent-aware contrastive learning을 적용하여, utterance와 intent name을 연결할 수 있는 Pre-trained Intent-aware Encoder(PIE)를 제안함.

## Introduction

Task Oriented Dialogue(TOD) 시스템에서 의도(intent) 분류의 label을 추가하기 위해서는 필요한 데이터를 수집하고, 모델을 재학습하는 과정이 필요하다. 최근 연구에서는 intent label의 의미를 활용해 zero-shot 또는 few-shot으로 분류를 하는 연구가 이뤄진다.

transformer 기반 text encoder를 사용해 class representation을 만들고, 주어진 query에 대해서 similarity metric을 사용해 class를 선택하는 Prototypical Network가 좋은 성능을 보인다. 이 방법을 기반으로 few-shot learning 방법과 text encoder에 대한 연구가 진행되었다. 기존에 존재하는 sentence bert와 같은 text encoder는 intent classification 문제에서 utterance와 intent name 사이의 representation을 비슷하게 만들도록 설계되어 있지 않기 때문에 성능의 한계가 있다. Text encoder에서 utterance와 intent name을 연결시켜주면 이 문제가 해소될 수 있지만, 다양한 intent set에 대한 데이터를 얻기에는 비용이 많이 필요한 어려움이 있다.

### Proposed method

![그림1](/assets/images/PIAE/1.png "그림1"){: .align-center}  

이 논문에서는 Zero / Few shot IC task를 위한 새로운 pretraining 방법을 제안한다. query 문장(utterance)에서 intent와 연관된 단어나 구절을 찾는 IRL(intent role labeling) tagger를 사용한다. IRL tagger의 값을 조합하여 pseudo intent를 만들고, query 문장과 함께 contrastive learning 방법으로 인코더를 학습시키는데 사용한다. 제안하는 intent-aware contrastive learning 방법은 임베딩 공간에서 pseudo intent name과 utterance를 연결하는 것 외에도 text encoder가 주어진 query 문장에서 intent를 구분하는데 중요한 부분에 집중하게 한다.

## Prototypical Networks for Intent Classification

### Prototypical Network (Snell et al., 2017)

![그림2](/assets/images/PIAE/2.png "그림2"){: .align-center}  

### Prototypical Networks for Intent Classification

*episode*: K example, N intent class (1개의 *episode*에 K x N개 utterance 보유)

*support set*: 각 intent class에 대한 sample utterances

*prototype*: *support set*을 인코딩하여 평균을 낸 class representation

$$c_n = \frac{1}{K} \sum_{x_{n,i} \in S_n} f_{\phi}(x_{n,i})$$

Inference 과정에서는 utterance를 인코딩하고, N개의 prototype과 비교하여 가장 가까운 것을 찾는다. (Euclidean or Cosine similarity, …) 

N개의 intent가 있고, 각 intent마다 K개 utterance가 있다면 N-way K-shot으로 부른다.

Intent name을 추가적인 정보로 주는 방법도 존재한다. 이 방법을 사용하면 example utterance 없이도 분류가 가능해 zero-shot classficiation을 할 수 있다.

$$c_n^{label} = \frac{1}{K+1}[ \sum_{x_{n,i} \in S_n} f_{\phi}(x_{n,i}) + f_{\phi}(y_n)]$$

$$c_n^{label} = f_{\phi}(y_n)$$

## Pseudo Intent Name Generation

IC를 위한 text endocer를 pre-train하기 위해 다양한 intent name에 대한 학습이 필요하다. 그러나 intent name을 수집하기 위해서는 비용이 들기 때문에 IRL tagger를 사용해 pseudo intent name을 생성한다. IRL은 utterance에서 intent와 관련된 정보(Intent Role)를 찾는다.

### Intent Role

논문에서는 6개의 intent role을 정의한다.

*Action*: intent와 관련있는 주요 동작

*Argument*: intent의 해석에 중요한 역할을 하는 action의 요소 / entity / event

*Request*: 질문이나 정보 검색을 위한 동사

*Query*: 질문이나 정보 검색의 정답에 대한 기대값

*Slot*: intent 해석에는 크게 영향이 없는 추가적인 값

*Problem*: 문제가 되는 상태나 이벤트에 대한 설명, 주로 비명시적인 값

IRL tagger의 학습을 위해 Schema-Guided Dialogue 데이터셋에서 위의 방법으로 Intent Role 데이터를 만들었다.

![그림3](/assets/images/PIAE/3.png "그림3"){: .align-center}  

### IRL Tagger

Sequence tagging problem으로 보고 BIO scheme을 사용해 학습했다.

RoBERTa-base 모델을 사용했다.

![그림4](/assets/images/PIAE/4.png "그림4"){: .align-center}  

### Generating Pseudo Intents

Pseudo intent name을 만들기 위해 Intent Role을 모두 연결한다. IRL prediction에서 action, argument, query가 없는 utterance는 제외했다.

![그림5](/assets/images/PIAE/5.png "그림5"){: .align-center}  

## Intent-Aware Contrastive Learning

Utterance와 intent name이 비슷한 representation을 갖도록 text encoder를 학습한다. Intent-aware contrastive learning 방법을 제안하는데, utterance, gold intent, pseudo intent를 바탕으로 학습된다. 

논문에서는 InfoNCE loss를 사용한다. i번째 샘플 $$x_i$$와 입력과 짝이되는 $$y = <y_1, y_2, ..., y_N>$$에 대해서 아래의 식처럼 loss를 계산한다.

$$
l(x_i, y) = \frac{exp[sim(f_{\phi}(x_i), f_{\phi}(y_i))]}{\sum_k^N{exp[sim(f_{\phi}(x_i), f_{\phi}(y_k))]}}
$$

여기서 N은 batch size이고, batch에 포함되는 샘플 중 positive sample이 아닌 것은 negative sample로 취급한다.

논문에서는 세가지 타입의 positive 샘플을 제안하는데, 두개는 supervised 이고 하나는 semi-supervised이다. 

첫번째 supervised positive 쌍은 입력 utterance와 pre-training 데이터 셋에 존재하는 gold intent name이다.

$$
L_{gold\_intent} = -\frac{1}{N}\sum_i^Nl(x_i, y^{gold})
$$

두번째 supervised positive 쌍은 입력 utterance와 같은 gold intent name을 갖는 gold utterance 쌍이다.

$$
L_{gold\_utterance} = -\frac{1}{N}\sum_i^Nl(x_i, x^{gold})
$$

세번째 semi-supervised positive 쌍은 입력 utterance와 pseudo intent name 쌍이다.

$$
L_{pseudo} = -\frac{1}{N}\sum_i^Nl(x_i, y^{pseudo})
$$

최종적으로 loss는 아래 세가지 loss를 더해 계산한다.

$$
L= L_{gold\_intent}+L_{gold\_utterance}+\lambda L_{pseudo}
$$

## Experiment

### Experiment settings

few-shot IC task에서 비교 모델로는 ProtoNet과 ProtAugment는 fine-tuning method로 사용되었고, SBERT가 pre-trained text encoder로 사용되었다.

ProtoNet과 ProtAugment는 모두 BERT base 모델을 바탕으로 학습에 사용되는 데이터를 pre-train  이를 BERT TAPT로 구분한다.

Pretraining data로 아래와 같은 데이터를 사용했다. intent 간의 불균형을 완화하기 위해 하나의 intent에서 TOP, DSTC11-T2는 최대 1000개, SGD, MultiWOZ는 100개의 utterance만 사용했다. 

![그림6](/assets/images/PIAE/6.png "그림6"){: .align-center}  

Downstream dataset으로는 아래와 같은 데이터를 사용했다.

Bangking77은 은행도메인, HWU64는 home assistant 로봇에서 사용되는 21개의 도메인을 갖고 있고, Liu54는 Amazon Mechanical Turk에서 모은 데이터, Clinc150은 여행, small talk 같은 도메인의 데이터이다. 

![그림7](/assets/images/PIAE/7.png "그림7"){: .align-center}  

Pre-train 데이터와 Downstream 데이터의 intent name이 overlapping되는 양을 조사했다.

(train, valid, test 셋을 나누어 조사했어야 few-shot, zero-shot이 제대로 되는지 알 수 있는데 이와 관련된 내용이 없다)

![그림8](/assets/images/PIAE/8.png "그림8"){: .align-center}  

Pre-train에서 text encoder는 SBERT를 사용했다. intent name을 example로 사용하는 경우 L-을 앞에 붙였다.

### Results

BERT_tapt : BERT base 모델을 training utterance를 사용해 추가로 pre-training 시킨 것

L- : prototype 생성에 레이블 이름을 사용한 것

![그림9](/assets/images/PIAE/9.png "그림9"){: .align-center}  

5 way, K shot 결과

![그림10](/assets/images/PIAE/10.png "그림10"){: .align-center}  

N way, K shot 결과, 5 way 결과 보다 더 큰 성능 향상이 있었다.

## Analysis

### Pre-training Corpus Ablation

![그림11](/assets/images/PIAE/11.png "그림11"){: .align-center}  

L-SBERT를 바탕으로 pre-train 데이터에 대한 분석. TOP가 성능을 가장 많이 올린다. 

### Pre-training Loss Ablation

![그림12](/assets/images/PIAE/12.png "그림12"){: .align-center}  

pseudo loss가 없어지면 성능이 많이 떨어진다. pseudo loss가 중요한 역할을 하고 있다는 것을 알 수 있다.

### Varying K and N

![그림13](/assets/images/PIAE/13.png "그림13"){: .align-center}  

Baseline에 비해 제안 모델의 성능이 좋고, K가 작을때의 성능 차이가 가장 크다. N이 커질수록 성능 차이가 커진다. 이 결과를 통해 intent가 많고 utterance 예제가 적은 실제 TOD 시스템에 적용할 수 있다는 것을 보여준다.

### Impact of Overlapping Intents

![그림14](/assets/images/PIAE/14.png "그림14"){: .align-center}  

intent overlapping이 있는 데이터를 제외하고 성능을 측정해보았다. 오히려 제외할 때 성능이 올라가는 경우가 있다. 이는 pre-training 데이터의 bias 때문이라고 한다. 예를 들어 pre-training 데이터에 존재하는 “please play my favorite song”이라는 utterance는 “play music”이라는 intent를 가진다. 학습된 모델이 테스트 데이터 “that song is my favorite”를 “play music”으로 분류하는데 정답은 “music likeness”이다.

## Conclusions

제안 모델이 large class, small example 환경에서 robust하게 동작하는 것을 증명했다. 추후 연구로는 IRL, PIE 모델로 multi-label IC 연구나 out-of-scope detection 연구가 있을 것이다. 한계로는 IRL tagger가 problem label에서 아주 낮은 정확도를 가지는 점, pseudo intent를 만들 때 모든 IRL label을 동등하게 사용했다는 점이 있다.