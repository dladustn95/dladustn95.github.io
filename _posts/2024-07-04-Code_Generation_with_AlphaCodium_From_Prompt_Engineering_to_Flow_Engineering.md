---
title:  "Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering 논문 읽기"
excerpt: "Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, CodeGeneration, arxiv24]
date: 2024-07-04T18:00:00+09:00
last_modified_at: 2024-07-04T18:00:00+09:00
---
  
# Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering

## Abstract

코드 생성 문제는 일반적인 자연어 문제와 다릅니다. 이러한 문제는 대상 언어의 정확한 구문 일치, 해피 패스와 엣지 케이스 식별, 문제 사양의 수많은 작은 세부 사항에 대한 주의, 그리고 기타 코드 특유의 문제와 요구 사항을 다루어야 합니다. 따라서 자연어 생성에서 성공적인 많은 최적화와 트릭이 코드 작업에서는 효과적이지 않을 수 있습니다. 본 연구에서는 LLMs를 통한 코드 생성에 새로운 접근 방식을 제안합니다. 우리는 이를 AlphaCodium이라 부르며, 테스트 기반의 다단계 코드 중심 반복 흐름으로, 코드 문제에 대한 LLMs의 성능을 향상시킵니다. 우리는 AlphaCodium을 Codeforces와 같은 플랫폼의 경쟁 프로그래밍 문제를 포함하는 도전적인 코드 생성 데이터셋인 CodeContests에서 테스트했습니다. 제안된 흐름은 결과를 일관되게 그리고 크게 향상시켰습니다. 예를 들어, 검증 세트에서 GPT-4의 정확도(pass@5)는 잘 설계된 단일 직접 프롬프트에서 19%에서 AlphaCodium 흐름으로 44%로 증가했습니다. 우리는 본 연구에서 습득한 많은 원칙과 모범 사례가 일반적인 코드 생성 작업에 널리 적용될 수 있다고 믿습니다.

## Introduction

최근 LLM을 활용한 code generation은 간단한 프로그래밍 문제에 대해서 좋은 성능을 보이고 있다. 그러나 실제 환경에서는 더 애매모호하고 긴 자연어로 표현되는 문제 설명이 주어지고, 정답 코드에는 여러 디테일과 규칙을 포함해야 한다.

이 논문에서는 AlphaCodium을 소개하는데, 입출력을 보고 반복적으로 코드 생성을 수정하는 방법을 제안한다. 

AlphaCodium은 
1. 문제 분석, 테스트 추론과 같은 추가 정보를 생성하고  
2. AI가 생성한 풍부한 public-test 데이터를 사용한다.

이 논문에서 주장하는 핵심 내용은 정확한 풀이를 생성하는 것보다 추가 테스트 데이터를 생성하는 것이 쉽다는 것이다. 테스트 데이터를 생성하기 위해서는 문제에 대한 이해가 필요하지만, 전체 문제를 완전히 풀 필요는 없다. 또한 본 논문에서는 코드 생성에 도움을 주는 방법, 컨셉을 제안한다. 

## CodeContests Dataset

![그림1](/assets/images/CGAC/1.png "그림1"){: .align-center}  

Codeforces와 같은 경쟁력 있는 프로그래밍 플랫폼에서 모은 데이터셋이다. 코딩 테스트와 비슷하게 자연어로 구성된 문제 설명, open test set, 문제당 200개 이상의 private test set으로 이뤄져 있다.

10k개의 데이터가 존재하고, 107개의 validation, 165개의 test 데이터가 존재한다.  LLM은 일반적으로 작은 디테일에 집중하지 않고, 일반적인 풀이를 생성한다. 그러나 실제 환경의 코딩 문제는 아주 작은 디테일까지 만족시켜야 한다. CodeContests 데이터셋의 문제 설명은 정교하고, 길이가 길고, 작은 디테일까지 포함되어 있다.

CodeContests 데이터셋을 사용한 이전 연구로는 AlphaCode가 있다. AlphaCode는 아주 많은 수 (최대 1M)의 가능한 풀이를 생성한다. 이를 클러스터링하고, 아주 작은 수 (~10개)의 답이 선택되고 제출된다. 이 방법은 모델을 fine-tuning 해야하고, 많은 수의 풀이를 생성하기 위해 막대한 컴퓨팅 자원이 요구되어 실제 환경에 사용되기 어렵다. CodeChain은 sub-module기반의 self-revision을 포함하는 방법이다.

## The AlphaCodium Flow

CodeContest에서 single prompt 모델 또는 COT 모델은 좋은 결과를 얻지 못한다. 모델은 문제를 이해하는데 어려움을 겪으며, 잘못된 코드를 생성하거나, public test에서 통과하지만 private test에서는 통과하지 못하는 코드를 생성한다. 

다른 NLP task에서 사용하는 방법 대신 Code generation에서는 생성한 코드를 실행하고 다시 수정하는 반복적인 프로세스가 더 낫다는 것을 보였다.

AlphaCodium의 flow는 다음과 같다

pre-processing은 선형 flow이고 문제에 대해서 자연어로 추론한다.

code iteration은 반복되는 stage로 AlphaCodium이 코드를 생성하고, 실행하고 다시 수정하는 동작을 한다.

**Flow stages**

![그림2](/assets/images/CGAC/2.png "그림2"){: .align-center}  

Problem reflection: 

문제를 설명한다. bullet point를 사용하고, 문제의 목표, 입출력, 규칙, 다른 디테일한 요소를 포함한다.

Public tests reasoning

각각의 public test의 입력이 output으로 유도되는 이유를 설명한다.

Generate possible solutions

자연어로 2~3개의 가능한 풀이를 생성한다.

Rank solutions

가능한 풀이의 순위를 메기고, 정확도, 간단함, 강건함을 고려해 순위를 메긴다. (가장 효율적인 풀이를 선택할 필요는 없다)

Generate additional AI tests

6~8개의 test로 사용할 입출력 값을 생성한다. 기존의 public test에서 포함하지 못하는 case를 포함하려 한다.

Initial code solution

문제에 대한 initial code를 생성한다. 이 코드는 정답과 비슷해야하고, run-fix 반복을 통해 더 수정되어야 한다.

Iterate on public tests

생성된 코드를 public test 데이터를 바탕으로 실행하고, 실패 시 수정한다.

Iterate on AI-generated tests

AI-generated tests로 동작시키고, 실패 시 수정한다.

run-fix iteration

potential solution을 선택하고 이에 해당하는 코드를 생성한다. 생성된 코드를 public, AI test 데이터에 실행한다. 모든 test에 통과하거나 최대 횟수에 도달하면 반복을 멈춘다.

**Additional insights**

1. 쉬운 단계에서 어려운 단계로 진행한다. 현재 단계에서 얻은 지식과 통찰력이 더 어려운 단계를 진행하는 데  도움을 준다. 예를 들어, problem refrection은 더 어려운 단계인 generate possible solution에 활용될 수 있다. pre-processing 단계의 결과는 code iteration에서 활용될 수 있다.
2. AlphaCodium이 갖는 또 다른 핵심은 보다 많은 test 데이터를 생성하는 것이 정답 코드를 생성하는 것보다 쉽다는 것이다. test 데이터를 생성하는 것이 문제에 대한 이해가 필요하긴 하지만, 문제를 완전히 풀이하는 것이 아니라 일부만 풀어도 되기 때문이다. 
fig 1 b 그림에서 ai-test가 문제 해결의 범위를 늘려준다.
3. 또한 그림의 프로세스는 single LLM과 결합할 수 있으며, 프로세스에서의 각 단계는 한번의 LLM call로 통합할 수도 있다.

### Code-Oriented Design Concepts

**YAML Structured output**

yaml로 구조화된 출력은 복잡한 task를 code와 비슷한 형태로 풀어낼 수 있게 한다. 

Yaml과 Pydnatic class를 활용해 instruction을 만들었다.

![그림3](/assets/images/CGAC/3.png "그림3"){: .align-center}  

최신버전의 GPT 모델은 json 출력을 지원하지만 따옴표, \n 처리에서 Yaml 형식이 더 유리하다.

![그림4](/assets/images/CGAC/4.png "그림4"){: .align-center}  

**Semantic reasoning via bullet points analysis**

Bullet 포인트로 심층적인 분석이 가능해지고, 모델이 의미에 따라 섹션을 구분할 수 있게 해준다.

**LLM do better when generating a modular code**

LLM이 하나의 길이가 있는 함수를 만들게 하면 나쁜 결과가 나온다. 또한 수정하는 과정에서도 모델이 어떤 부분을 수정해야할 지 혼동하게 된다. 대신 이를 작은 sub-function으로 나누어 만들게 하면 더 나은 성능을 얻을 수 있다.

**Soft decisions with double validation**

모델에게 예/아니오 와 같은 답을 요구하기 보다 더 추론을 하게 만든다. 예를 들어 AI-generated test에서 만들어진 input-output 쌍을 검증할 때 예/아니오로 대답하게 하는 것보다, Input만 주고 output을 다시 생성하게 한 뒤 잘못된 출력이 있는 경우 이를 수정하도록 하는 것이 효과적이다.

**Postpone decisions, try to avoid direct questions, and leave room for exploration**

복잡한 문제에 대해서 모델에게 직접 질문하기보다 쉬운 것부터 어려운 작업까지 점진적으로 질문한다.

- 가장 쉬운 것부터 시작: 주어진 문제에 대한 추론
- AI test 생성, 가능한 풀이 생성
- 코드 생성 및 run-fix 반복 시작

**Test anchors**

AI test 셋에서 실패할 때 생성한 데이터의 문제인지 코드의 문제인지 알아야 함.

- 틀리지 않은 것을 알고 있는 public test 데이터셋과 정답을 맞춘 AI test 데이터셋을 ‘anchor’ 데이터 셋으로 둔다
- AI test 데이터에서 틀린 것이 발생했을 때 코드에 문제가 있는 것으로 간주하고 코드를 수정한다.
- 수정된 코드를 검증할 때 반드시 anchor 데이터 셋 (이전에 맞춘 데이터 셋)을 모두 통과해야 한다.

### Results

**Direct prompt vs. AlphaCodium flow**

![그림5](/assets/images/CGAC/5.png "그림5"){: .align-center}  

풀이 코드를 생성하게 하는 prompt와 AlphaCodium의 성능 비교

pass@5는 문제마다 5개의 풀이 코드를 생성했을 때 그 안에 정답이 있는 경우를 계산한 것이다. 

**Comparison to previous works**

![그림6](/assets/images/CGAC/6.png "그림6"){: .align-center}  

이전 연구인 CodeChain과 AlphaCode와 성능 비교.

AlphaCode는 100K개의 풀이를 만든 뒤 클러스터링해서 대표값 10개만 사용하는 방식

더 적은 컴퓨팅 자원으로 더 좋은 성능을 보인다. 

**Computational effort and comparison to AlphaCode and AlphaCode2**

AlphaCodium은 하나의 풀이 당 15~20 LLM 호출을 하고, pass@5를 적용하면 약 100번의 호출을 한다.

AlphaCode는 100K개 풀이 생성을 위해 적어도 100K개의 LLM 호출이 있어야한다. 

AlphaCode2는 Gemini-Pro를 fine-tuning 했고, 100개의 풀이를 생성하여 AlphaCode와 비슷한 결과를 얻었다고 했다.

### Conclusions

반복해서 코드를 동작시키고 수정하는 방식의 AlphaCodium 모델을 제안했다. 

Flow는 문제에 대해 자연어로 추론하는 pre-processing 단계와 코드를 생성하여 실행하고, 수정하는 run-fix 단계가 있다. 

AlphaCodium은 코드 생성에서 성능 향상에 도움이 되는 기법을 찾아서 적용했다.(YAML 포맷 사용, modular code 생성, bullet point로 추론 등)

제안 모델은 CodeContests 데이터셋에서 좋은 성능을 보였다.