---
title:  "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes 논문 읽기"
excerpt: "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, ACL23]
date: 2023-12-05T18:00:00+09:00
last_modified_at: 2023-12-05T18:00:00+09:00
---
  
# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

## Abstract

LLM(대규모 언어 모델) 배포는 실제 애플리케이션에 비해 메모리가 비효율적이고 컴퓨팅 집약적이기 때문에 어렵습니다. 이에 대응하여 연구자들은 사람이 작업한 레이블로 미세 조정하거나 LLM이 생성한 레이블을 사용하여 증류하여 더 작은 task-specific 모델을 학습시킵니다. 그러나 미세 조정이나 증류 방법이 LLM과 비슷한 성능을 달성하기 위해 많은 양의 학습 데이터가 필요합니다. (a) LLM보다 성능이 뛰어난 더 작은 모델을 훈련하고 (b) 미세 조정이나 증류에 필요한 더 적은 양의 훈련 데이터를 활용하여 이를 달성하는 새로운 메커니즘인 Distilling step-by-step을 소개합니다. 우리의 방법은 multi-task 프레임워크 내에서 작은 모델을 교육하기 위한 추가 정보로 사용하기 위해 LLM의 추론 내용을 추출합니다. 우리는 4개의 NLP 벤치마크에서 세 가지 결과를 제시합니다. 첫째, 미세 조정 및 증류와 비교할 때 우리 메커니즘은 훨씬 적은 수의 레이블/레이블이 없는 훈련 예제로 더 나은 성능을 달성합니다. 둘째, few-shot prompt LLM에 비해 훨씬 더 작은 모델 크기를 사용하여 더 나은 성능을 달성합니다. 셋째, LLM을 능가하는 데 필요한 모델 크기와 데이터 양을 모두 줄입니다. 미세 조정된 770M T5 모델은 벤치마크에서 사용 가능한 데이터의 80%만을 사용하여 few-shot prompted 540B PaLM 모델보다 성능이 뛰어납니다. 그러나 동일한 크기의 T5 모델을 일반적인 방법으로 미세 조정하면 데이터 100%를 사용하더라도 동일한 성능을 내는 것이 어렵습니다.

## TL; DR

- LLM은 사용하기 위해 많은 컴퓨팅 자원을 요구하며, 이를 해결하기 위해 작은 모델을 학습시키는 방법을 사용함
- 데이터의 레이블과 추론 근거를 동시에 학습시키는 multi-task learning을 사용해 작은 모델이 적은 수의 데이터로 LLM prompt보다 더 나은 성능을 갖게 함
- 추론 근거와 레이블을 동시에 생성하지 않고 따로 따로 생성이 가능해 생성 속도의 향상이 가능함

## Introduction

![그림1](/assets/images/DSS/1.png "그림1"){: .align-center}  

175B ~ 500B 크기의 LLM을 배포하여 서비스에 사용하는 것은 컴퓨팅 자원 비용 문제 때문에 어려움이 존재한다. 

이를 해결하기 위해 finetuning이나 distillation으로 학습한 작은 task-specific 모델을 대신 사용하였다. 그러나 사람이 작업한 데이터가 필요하거나 (finetuning), LLM이 생성한 많은 양의 unlabeled data가 필요하다 (distillation). 이 방법 역시 데이터를 생성하기 위한 비용 문제가 있다.

적은 데이터로 작은 모델을 학습하는 새로운 메커니즘, Distilling step-by-step 방법을 제안한다.

LLM이 label을 예측할 때 추론 근거를 생성할 수 있다는 점을 주목했다. LLM이 생성한 추론 근거에는 작은 task-specific 모델이 배워야하는 task 관련 지식이 포함될 수 있다. 추론 근거를 추가적인 정보로 활용하여 label 및 추론 근거 예측을 모두 사용하는 multi-task learning을 통해 작은 모델을 학습한다.

## Distilling step-by-step

![그림2](/assets/images/DSS/2.png "그림2"){: .align-center}  

- unlabeled 데이터가 주어지면 LLM을 사용해 추론 근거와 레이블을 생성한다.
- 이 두개를 작은 모델에 학습시킨다.
- 추론 근거는 입력이 왜 해당 레이블로 매핑되어야 하는지에 대한 풍부하고 다양한 정보를 준다.

![그림3](/assets/images/DSS/3.png "그림3"){: .align-center}  

1. 추론 근거를 얻기 위해 CoT 방법을 사용했다.  
unlabeled dataset $$x_i \in D$$ 에 대해서 프롬프트를 사용해 추론 근거 $$\hat{r}_i$$와 정답 $$\hat{y}_i$$을 생성한다. 
2. 추론 근거를 그대로 입력으로 사용하면 항상 LLM이 생성한 추론 근거를 필요로 하기 때문에 작은 모델을 사용하는 이유가 없어진다.  
small model $$f$$에 대해서 $$f(x_i, \hat{r}_i) = y_i$$ 의 형태가 되면 안된다.  
3. 추론 근거를 입력으로 사용하는 대신 학습을 시켜 multi-task 문제로 변환했다.  
$$f(x_i) = (y_i, r_i)$$ 의 형태로 label과 추론 근거를 모두 생성할 수 있게 했다.  
$$L = L_{label}\ +\ \lambda L_{rationale}$$  
$$L_{label} = \frac{1}{N}\sum_{i=1}^{N}\ell(f(x_i), \hat{y}_i)$$  
$$L_{rationale} = \frac{1}{N}\sum_{i=1}^{N}\ell(f(x_i), \hat{r}_i)$$  
레이블과 추론 근거 각각은 task prefix를 사용해 생성을 구분했다. ([label], [rationale])  

# Experiments

### Setup

540B PaLM을 LLM으로 사용

T5를 task-specific downstram model로 사용

NLI 데이터, QA데이터, 수학 문제 데이터 사용

### Reducing training data

At labeled data

![그림4](/assets/images/DSS/4.png "그림4"){: .align-center}  

At unlabeled data

![그림5](/assets/images/DSS/5.png "그림5"){: .align-center}  

### Reducing model size

At labeled data

![그림6](/assets/images/DSS/6.png "그림6"){: .align-center}  

At unlabeled data

![그림7](/assets/images/DSS/7.png "그림7"){: .align-center}  

SVAMP에서는 성능이 안좋은데, 이것은 데이터 수 부족 (800개)의 문제로 보고, data augmentation을 했다. 2305개의 ASDiv 데이터를 추가로 사용해서 실험을 진행했다.

### Outperforming LLMs using minimum model size and least training data

At labeled data

![그림8](/assets/images/DSS/8.png "그림8"){: .align-center}  

At unlabeled data

![그림9](/assets/images/DSS/9.png "그림9"){: .align-center}  

### Ablation Study

![그림10](/assets/images/DSS/10.png "그림10"){: .align-center}  

LLM 모델의 크기가 클수록 성능이 좋다.

![그림11](/assets/images/DSS/11.png "그림11"){: .align-center}  

추론 근거와 정답을 합쳐서 single-task로 학습하는 것보다 multi-task가 성능 향상에 더 도움이 된다.

$$L = \frac{1}{N}\sum_{i=1}^{N}\ell(f(x_i), [\hat{r}_i, \hat{y}_i])$$ → sinlge-task training

근거~~~, 정답: 1

또한 single-task로 학습하면 생성 과정에서 추론 근거를 모두 생성해야 하기 때문에 시간이 오래 걸린다. 

# Discussion

- 이 논문에서는 추론 근거를 얻기 위해 few-shot CoT를 사용했고, 사용자가 직접 추론 내용을 만들어야 하는 단점이 존재한다. 그러나 최근 연구에서 zero-shot CoT가 가능하다는 것이 알려져서 이 점이 보완될 수 있게됐다.
- 학습 과정에서 추론 근거와 레이블을 동시에 학습시키는 것이 computational overhead를 일으킨다.
- 추론 근거의 품질이 성능에 끼치는 영향을 조사할 필요가 있다.