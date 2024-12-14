---
title:  "Agent-as-a-Judge_Evaluate_Agents_with_Agents 논문 읽기"
excerpt: "Agent-as-a-Judge_Evaluate_Agents_with_Agents 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, arxiv24]
date: 2024-10-10T18:00:00+09:00
last_modified_at: 2024-10-10T18:00:00+09:00
---
  
# Agent-as-a-Judge_Evaluate_Agents_with_Agents

## Abstract

현대의 평가 기법은 에이전트 시스템에 적합하지 않습니다. 기존 접근 방식은 결과에만 집중하거나, 에이전트 시스템의 단계별 특성을 무시하며 지나치게 많은 수작업을 요구합니다. 이를 해결하기 위해, 우리는 Agent-as-a-Judge 프레임워크를 도입합니다. 이 프레임워크는 에이전트 시스템을 사용하여 에이전트 시스템을 평가하는 방식으로, LLM-as-a-Judge 프레임워크의 유기적 확장이며, 전체 문제 해결 과정에 대해 중간 피드백을 제공할 수 있는 에이전트 특성을 통합합니다. 우리는 Agent-as-a-Judge를 코드 생성 작업에 적용합니다. 기존 벤치마크의 문제를 극복하고 Agent-as-a-Judge에 대한 개념 증명 시험대를 제공하기 위해, 55개의 현실적인 자동화된 AI 개발 작업을 포함하는 새로운 벤치마크인 DevAI를 제시합니다. 이 벤치마크는 365개의 계층적 사용자 요구 사항을 포함하는 풍부한 수동 주석을 제공합니다. 우리는 Agent-as-a-Judge를 사용하여 인기 있는 세 가지 에이전트 시스템을 벤치마킹한 결과, LLM-as-a-Judge를 크게 능가하며 인간 평가 기준만큼 신뢰할 수 있음을 확인했습니다. 전반적으로 우리는 Agent-as-a-Judge가 동적이고 확장 가능한 자기 개선을 위한 풍부하고 신뢰할 수 있는 보상 신호를 제공함으로써 현대 에이전트 시스템을 위한 중요한 진전을 나타낸다고 믿습니다.

## Introduction

![그림1](/assets/images/AAAJ/1.png "그림1"){: .align-center}  

agent 시스템을 평가하는데 가장 큰 이슈는 대부분 마지막 결과로만 평가하고, 문제를 푸는 과정을 평가하지 못하는 것이다.

agent 시스템은 사람처럼 step-by-step으로 생각하고 동작하기 때문에 사람과 전체 thought and action 기록을 보고 비슷하게 평가해야한다. 

그러나 사람이 이를 평가하는 것은 너무 비싸기 때문에 본 논문에서는 LLM-as-a-Judge에서 영감을 받아 agent 시스템을 평가하는 agent 시스템인 Agent-as-a-Judge를 제안한다.

이 방법은 LLM-as-a-Judge의 비용 효율성 뿐만 아니라 중간 피드백을 제공할 수 있는 agent의 특성도 갖고 있다.

이 논문에서는 code generation task에 집중하여 실험을 진행한다.

현재 존재하는 code generation 데이터 셋은 실용적인 문제를 다루지 못하고 있고, Code generation 분야에서 LLM은 이미 benchmark의 27% 이상을 agent 시스템 없이 해결할 수 있다. 특히 최근 연구들은 특정 데이터셋에 특별히 설계된 메커니즘을 도입하기 시작했다. 

이러한 문제를 해결하기 위해 본 논문에서는 real-world에서 쓰이는 55개의 AI 관련 code generation 데이터셋을 공개한다.

또한 평가를 위해 3개의 agentic framework를 사용했다. (MetaGPT, GPT-Pilot, OpenHands)

- 55개의 종합적인 AI 개발 작업으로 구성된 DevAI 데이터셋을 출시하며, 여기에는 태그, 개별 계층적 요구 사항 및 개별 선호 사항이 포함된다.
- 세 가지 주요 오픈 소스 코드 생성 에이전트 프레임워크로 실험하고, 기존의 평가보다 더 포괄적인 분석을 제공한다.
- Agent-as-a-Judge 개념을 소개한다.
- Agent-as-a-Judge가 LLM-as-a-Judge보다 뛰어나며, 인간 평가자와 비슷한 성과를 낸다는 것을 보인다.

## DevAI: A Dataset for Automated AI Development

### Motivation

현재 존재하는 code generation 데이터셋은 사용자의 자연어로 된 요청에서부터 코드의 완성까지 실제 개발 과정의 end-to-end가 아닌 일부만을 다루고 있다. (SWE-Bench automated repair / HumanEval algorithm)

또한 개발 과정의 중간 단계를 평가하지 못하고 결과물만을 평가한다 (pass@1)

### The DevAI Dataset

![그림2](/assets/images/AAAJ/2.png "그림2"){: .align-center}  

![그림3](/assets/images/AAAJ/3.png "그림3"){: .align-center}  

개발 작업을 설명하는 query, 총 365개의 요구사항을 포함하는 requirements, 총 125개의 선호사항을 설명하는 preferences로 이뤄진다.

DevAI는 AI와 관련된 문제이고, 각 문제마다 tag가 달려있다.

요구사항은 DAG(directed acyclic graph) 형태로 배열된다. 

### Preliminary Benchmark

논문에서는 이 데이터셋에 대해 3개의 agent를 사용해 benchmark 성능을 만들었다. 

agent 시스템의 backbone LLM으로 gpt4o를 사용했고, 시간제한 1800초를 주고 이 시간이 지나면 강제 종료했다. 

![그림4](/assets/images/AAAJ/4.png "그림4"){: .align-center}  

## Human-as-a-Judge:Manual Evaluation on DevAI

### Benchmark Baselines by Human-as-a-Judge

3명의 인간 평가자에게 AI developer의 결과물을 평가하게 했다.

Human evaluation에 나타나는 bias를 얻기 위해 첫번째 평가 단계에서는 최소한의 지침을 주고, 평가자들이 평가 기준에 대해 논의하게 했다. 

첫번째 단계 이후 각 평가자들에게 토론을 통해 결과에 대한 합의를 보도록 했고, 이 결과를 최종 평가 결과로 사용했다. (오랜 시간을 들여 최대한 공정한 평가를 하도록 함)

**Performance Analysis**

![그림5](/assets/images/AAAJ/5.png "그림5"){: .align-center}  

GPT-Pilot, OpenHands가 가장 성능이 높은데도 29%의 요구사항만 충족한다. 

즉 DevAI 데이터셋이 상당히 어렵다. 

평가자들이 요구사항을 만족하는지 평가하는 항목을 추가한 것은 우리의 기대와 동일하다. 

agent 시스템이 문제를 풀이하는 과정을 평가하는 것이 최종 결과만 평가하는 것보다 더 풍부한 피드백을 줄 수 있다.

### Judging Human-as-a-Judge

![그림6](/assets/images/AAAJ/6.png "그림6"){: .align-center}  

Human as a judge의 신뢰성과 bias의 존재를 분석하기 위해 불일치도를 측정했다.

불일치도는 10~30% 정도로 나타났다.

이는 여러 단계별로 동작하는 AI developer가 동작하는 방식의 복잡성 때문이기도 하고, 일부는 모호한 부분에 대한 관점의 차이이기도 하다.

앞서 언급했듯 평가자는 첫번째 개별 평가 이후 토론을 통해 합의를 도출한 평가를 만드는 과정을 진행했다.

Human as a judge 파이프라인에서 평가자는 다른 평가자의 근거를 수용함으로써 자신의 판단 오류를 인정하고 평가를 조정할 수 있다.

![그림7](/assets/images/AAAJ/7.png "그림7"){: .align-center}  

합의 평가가 실제 참 값과 동일하거나 동일하지 않을 수 있지만, 다수결로 평가를 선택하는 것이 개별적으로 평가하는 것보다 더 합의 평가에 근접한 결과를 얻을 수 있다.

결론적으로 사람의 판단 오류는 불가피하고, 이를 줄이기 위해 본 논문에서는 2가지 방법을 제안한다.

1. 개별적으로 판단한 이후에 토론을 통해 도출된 평가를 만들도록 한다. (소수로도 할 수 있다.)
2. 정확도가 50% 이상인 전문적인 평가자를 여러명 구성하는 것 (많을수록 좋지만 비용이 들어간다)

## Agent-as-a-Judge: Evaluating Agents with Agents

### Proof-of-Concept

![그림8](/assets/images/AAAJ/8.png "그림8"){: .align-center}  

(1) Graph는 파일, 모듈, 의존성을 포함하는 프로젝트의 전체 구조를 그래프로 구성한다. 

(2) Locate는 요구 사항에서 참조된 특정 폴더나 파일을 식별한다. 

(3) Search는 코드의 context를 이해하여 제공한다. 또한 관련 있는 code snippets과 그 안의 dependency와 같은 뉘앙스를 빠르게 retrieve한다.

(4) Retrieve는 긴 텍스트에서 정보를 추출하여 궤적 내 관련 세그먼트를 식별합니다. 

(5) Read는 코드, 이미지, 비디오, 문서를 포함한 33가지 다양한 형식의 멀티모달 데이터를 읽고 이해한다. 이를 통해 에이전트가 다양한 데이터 를 교차 참조하고 다양한 요구 사항을 검증할 수 있게 한다. 

(6) ASK는 위의 정보를 바탕으로 주어진 요구 사항이 충족되는지 여부를 결정한다.

(7) 메모리 모듈은 판단 정보를 기록한다. 

(8) 계획 모듈은 현재 상태와 프로젝트 목표를 기반으로작업을 전략화하고 순서를 계획한다.

![그림9](/assets/images/AAAJ/9.png "그림9"){: .align-center}  

ablation study 결과 graph, locate, read, retrieve, ask 모듈의 조합이 가장 좋은 성능을 보였다.

논문에서는 그 이유로 Agent-as-a-judge가 높은 퀄리티의 정보를 필요로하고, noise에 민감하기 때문으로 가정했다. 

예를 들어 Planning 모듈이 미래의 action에 대한 의사 결정을 내려줄 것으로 기대했지만 이 과정이 불안정했다.   
먼저 memory 모듈의 기록이 도움이 될 것으로 기대했지만, 과거의 오류가 연쇄작용을 일으키는 문제가 있었다.   
또한 agent가 생성한 수백줄 밖에 안되는 코드는 search 모듈의 도움을 받기에는 너무 적었다.  

## Judging Agent-as-a-Judge and LLM-as-a-Judge

![그림10](/assets/images/AAAJ/10.png "그림10"){: .align-center}  

비교 결과가 얼마나 차이나는지 측정하는 Judge shift.

Human-as-a-Judge 결과와 비교할 때 Agent-as-a-Judge가 일관되게 적은 차이를 보인다. 

이는 Agent-as-a-Judge가 작업 요구사항을 충족시키기 위한 안정성과  적합성이 높다는 것을 보여준다.

Alignment rate는 전체 365개의 요구사항에 대해 사람과 얼마나 유사하게 평가했는지 보여준다. 

복잡한 시나리오일수록 Agent-as-a-Judge가 사람과 더 유사하게 평가한다.

![그림11](/assets/images/AAAJ/11.png "그림11"){: .align-center}  

Agent를 평가하는 과정은 불균형 작업으로, 요구사항을 실패하는 경우가 너무 많다. (MetaGPT는 요구사항을 거의 충족하지 않는다. LLM-as-a-Judge는 대부분을 부정으로 평가하고 alignment rate 84.15%가 나옴)

따라서 PR curve로 평가를 다시 했는데, Agent-as-a-judge가 사람과 거의 비슷한 성능을 보임

### Ablations For Agent-as-a-Judge

![그림12](/assets/images/AAAJ/12.png "그림12"){: .align-center}  

각각의 모듈을 추가할 때 성능 변화. 제외한 것을 넣으면 성능이 하락한다.

retrieve는 여기서는 큰 변화가 없지만, MetaGPT와 GPT-Pilot에는 retrieve가 유용하다.

### Cost Analysis

Human-as-a-Judge는 3명의 평가자가 총 86.5시간을 소모했고, 평가자의 시급을 15달러로 가정하면 총 1297.5 달러가 소모되었다.

Agent-as-a-Judge는 API 호출 비용 30.58 달러가 들었고, 소요 시간은 118.43분이 걸렸다.

LLM-as-a-Judge는 10.99분이 걸렸고, 29.63 달러가 들었다. 

## Discussion and Conclusion

**Outlook 1: Intermediate Feedback for Agentic Self-Improvement**

Agent-as-a-Judge의 주요 강점은 중간 피드백을 통해 효과적이고 효율적인 최적화가 가능하다는 점이다. 에이전트 시스템이 이를 사용하여 복잡한 다단계 문제에 대한 해결책에서 문제를 실시간으로 식별하고 수정할 수 있다. 따라서 agent system을 향상시키기 위해 process-supervised reward model (PRM)을 만들 수 있을 것이다.

**Outlook 2: Flywheel Effect Driven by Agent-as-a-Judge**

Agent-as-a-Judge와 피평가자(agent)는 반복적인 피드백을 통해 상호 개선되는 cycle을 만들 수 있다. 따라서 Agent-as-a-Judge를 핵심 기술로 사용하여 self-play system을 agent 버전으로  구축할 수 있을 것이다.

점진적인 개선이 지속되면 점점 더 큰 최적화와 성능 향상을 이끌어낼 것이다. 이 반복적인 과정은 LLM 추론 데이터를 보완하는 귀중한 역할을 할 수 있으며, 에이전트 능력을 foundation 모델에 통합하는 데 도움을 줄 수 있다.

**Conclusion**

본 논문에서는 Agent-as-a-Judge 방법을 소개하고, DevAI 데이터 셋을 공개했다.

Agent-as-a-Judge가 기존 방법을 능가하고, human evaluator의 앙상블과 비슷한 성능을 보인다.