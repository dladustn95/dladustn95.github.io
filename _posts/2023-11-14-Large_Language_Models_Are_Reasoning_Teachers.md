---
title:  "Large Language Models Are Reasoning Teachers 논문 읽기"
excerpt: "Large Language Models Are Reasoning Teachers 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, ACL23]
date: 2023-11-14T18:00:00+09:00
last_modified_at: 2023-11-14T18:00:00+09:00
---
  
# Large Language Models Are Reasoning Teachers

## Abstract

최근 연구들은 Chain-of-Thought (CoT) 프롬프트가 언어 모델이 복잡한 추론 작업을 단계별로 수행하여 해결하게 할 수 있다는 것을 보였다. 그러나 CoT 방법은 GPT-3 175B처럼 대규모 배포가 불가능한 매우 큰 모델에 의존적인 문제가 있다. 이 논문에서는 큰 모델을 *reasoning teacher*로 사용하여 작은 모델에서 복잡한 추론을 가능하게 하고, 모델의 크기를 몇 배나 줄일 수 있다. 이 논문에서는 아주 큰 모델에서 추론 샘플을 생성하여 작은 모델에 미세 조정을 하는 *Fine-tune-CoT* 방법을 제안한다. 다양한 공개 모델과 여러 복잡한 작업에 대해 성능을 평가하였다. Fine-tune CoT는 작은 모델이 상당한 추론 능력을 가지게 하며, 많은 작업에서 프롬프트 기반 baseline의 성능을 뛰어넘고, 교사 모델의 성능 또한 능가한다. 또한 교사 모델이 각 원본 샘플에 대해 다양한 추론 근거를 생성할 수 있는 능력을 활용해 Fine-tune CoT 방법을 확장한다.  교사 모델의 diverse reasoning 결과를 활용한 풍부한 미세 조정 데이터는 아주 작은 모델에서도 평가 데이터셋 전반에서 상당한 성능 향상이 일어나게 한다. 학생 모델의 추론 능력의 출현을 이해하기 위해 연구를 수행하고 ablation 및 샘플 연구를 실시하다.

## TL; DR

- CoT는 복잡한 문제를 순서대로 추론하여 해결할 수 있으나 100B 단위의 LLM에서만 잘 동작한다는 단점이 있음
    - 컴퓨팅 자원과 추론 비용의 한계로 인해 LLM의 대규모 배포는 어려운 상황
- 작은 크기의 모델을 LLM이 생성한 CoT 근거와 함께 finetuning하는 방법을 제안
    - LLM의 추론 능력을 활용해 작은 모델에게 복잡한 문제를 푸는 방법을 가르침
    - 더 작은 모델로 teacher 모델보다 더 좋은 성능을 얻을 수 있음
- 하나의 샘플에 대해 여러 근거를 생성하는 augmentation 방법으로 추가적인 성능 향상을 얻음

## Introduction

![그림1](/assets/images/LLMART/1.png "그림1"){: .align-center}

## Chain-of-Thought Fine-Tuning

1. LLM teacher에 Zero-shot-CoT를 사용해 추론 근거와 LLM이 예측한 답을 생성한다.  
샘플 $$S_i$$가 각각 질문 $$q_i$$, 정답 $$a_i$$를 갖고 있을 때, 
Zero-shot-CoT를 이용해 추론 근거 $$\hat{r}_i$$와 예측한 답 $$\hat{a}_i$$를 생성한다.  
프롬프트와 생성 결과를 포함한 포맷은 아래와 같다  
Q: <$$q_i$$>. A: Let’s think step by step. <$$\hat{r}_i$$> Therefore, the answer is <$$\hat{a}_i$$>
2. 생성 결과를 필터링하고, 이 를 prompt-completion 쌍으로 재구성한다.  
$$a_i$$와 $$\hat{a}_i$$를 비교하여 같은 것만 사용하는데, 이로 인해 학습 샘플 수의 감소가 일어난다.  
($$S_i$$, $$\hat{r}_i$$, $$\hat{a}_i$$)를 $$S^{\prime}_i=(p_i,c_i)$$ 의 형태로 재구성한다.   
$$p_i$$ = “<$$q_i$$> ###”       $$c_i$$ = “<$$\hat{r}_i$$> - -> <$$\hat{a}_i$$> [END]”  
LLM teacher가 예측한 정답이 맞더라도 추론 근거가 틀리는 경우가 있는데, Appendix에서 추론 근거까지 정답인 것을 선택하는 것보다 추론이 틀리더라도 더 많은 데이터를 사용하는 것이 성능에 유리하다는 실험 결과를 제시했다.  
![그림2](/assets/images/LLMART/2.png "그림2"){: .align-center}

3. 수집한 데이터로 작은 사전학습 언어모델을 fine-tune한다.  
사전 학습에 사용한 training objective를 그대로 사용한다.  
**Diverse reasoning**  
Fine-tune-CoT의 teaching 효과 극대화를 위해 하나의 sample에서 여러 추론 근거를 생성한다.  
Teacher 모델에서 greedy decoding을 하는 대신 stochastic sampling 방법을 사용해 여러 가지 추론 근거를 생성할 수 있다.  
Greedy decoding: $$(\hat{r}_i, \hat{a}_i)$$  
Stochastic sampling: $$\{(\hat{r}_{ij}, \hat{a}_{ij})\}^D_j$$,  D = degree of reasoning diversity

## Experiments

### **Tasks and datasets**

수학 계산: SingleEq, AddSub, MultiArith, GSM8K, SVAMP, Aqua

날짜 계산: Date Understanding

물건 옮기기: Tracking Shuffled Objects

기호 관련: Last Letter Concatenation, Coin Flip

### Results

![그림3](/assets/images/LLMART/3.png "그림3"){: .align-center}

Fine-tune-CoT elicits complex reasoning in small models

Small models can outperform very large teachers in reasoning

![그림4](/assets/images/LLMART/4.png "그림4"){: .align-center}

Diverse reasoning substantially improves Fine-tune-CoT performance

![그림5](/assets/images/LLMART/5.png "그림5"){: .align-center}

Fine-tune-CoT consistently benefits from more data

![그림6](/assets/images/LLMART/6.png "그림6"){: .align-center}

Better reasoners are better teachers

![그림7](/assets/images/LLMART/7.png "그림7"){: .align-center}

Fine-tune-CoT performance scales with model size for small LMs

### Sample study

- 숫자 계산 문제에서는 계산 오류가 자주 일어나지만, diverse reasoning을 통해 많이 감소시킬 수 있다.
- 어려운 task인 GSM8K와 Aqua에서는 좋은 성능을 보이진 못했다.
- 제안 방법은 text 기반 task에서 뛰어남을 보였다.
- Zero-shot-CoT student는 질문을 반복하거나 일관되지 않고, 반복적인 응답을 생성하는 경우가 많다.
- Few-showt-CoT student는 단계적으로 추론 근거를 생성하지만 질문을 이해하지 못한 것처럼 보이고, 논리적, 상식적 오류가 포함되는 경우가 있다.

### Nuances of fine-tuning on CoT reasoning

- 정답이 맞지만 추론 근거가 틀린 경우가 있다.
    - 틀린 추론 근거도 학습에 도움이 된다.
- 기존 max length 때문에 CoT 생성 중 끊기는 경우가 있다.
    - max length를 늘려주면 성능 향상이 가능하다.

## Discussion

- 제안 방법은 low-resource 환경에서도 수행 가능하기 때문에 접근성이 좋다.
- 제안 방법이 diverse reasoning, 데이터 크기, 교사 모델 성능, 학생 모델 크기 등 여러 방면으로 확장될 수 있다.
- Fine-tune-CoT의 trade-off
    - diverse reasoning을 많이 하면 성능이 좋아지지만, 비용이 증가한다.
    - 그러나 diverse reasoning은 fine-tune 데이터를 만드는 것에 비해 저렴하다
- 지도학습이 작은 모델의 추론적 오류를 점진적으로 감소시킨다.
    - zero-shot, few-shot, fine-tune-CoT에서 추론 능력이 확연히 구분됨
    - 계산 오류나, 질문 이해와 같은 의미론적 오류가 감소함
- 주어진 task에 대해 더 큰 교사 모델을 연상시키는 추론 능력을 보임
    - 문제 해결을 위해 의미와 추론 단서를 인식하고, 큰 문제를 나누어 해결하는 능력을 가짐
    - 그러나 작은 도메인에서만 동작하고, 더 큰 지식이 필요한 task에서는 잘 동작하지 않을 것이라고 가정함
- 제안 방법의 distillation 능력으로, 미래에 LLM이 성능이 더 발전하면 작은 모델의 성능도 향상될 가능성이 있음

## Limitation

- 간결한 답변을 생성하게 하는 것도 필요하다.
    - Student 모델의 추론 근거가 반복적일 때가 존재하는데, 이건 추론 시간의 관점에서 비효율적이다.
    - teacher 모델의 추론 근거가 짧아지면, student 모델 역시 짧은 추론 근거를 생성한다.
- 다양한 범위의 모델을 사용해야 한다.
    - 실험에 사용한 모델이 SOTA는 아니었다.
    - 좋은 teacher 모델을 사용하거나, 좋은 student 모델을 사용하면 성능 향상의 여지가 있다.
- Zero-shot-CoT를 사용해 학습 샘플을 만들었는데, Few-shot-CoT를 사용해서 만들면 성능 향상이 더 있을 것이다.
- KD와 일부 특성을 공유한다.
    - dark knowledge가 KD에 존재하는 것처럼, 많은 양의 조금은 부정확한 sample이 적은 수의 완벽한 샘플보다 좋을 때도 있다.
    - 반면 KD는 좋은 teacher 모델이 항상 좋은 student를 만드는 것은 아니지만, Fine-tune-CoT는 이 것을 보장한다.
    - 향후 Fine-tune-CoT와 전통적인 KD 방법을 결합하는 연구도 가능해보인다.