---
title:  "AdaPlanner: Adaptive Planning from Feedback with Language Models 논문 읽기"
excerpt: "AdaPlanner: Adaptive Planning from Feedback with Language Models 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, AcionPlannig, NeurIPS23]
date: 2024-07-04T18:00:00+09:00
last_modified_at: 2024-07-04T18:00:00+09:00
---
  
# AdaPlanner: Adaptive Planning from Feedback with Language Models

## Abstract

대규모 언어 모델(LLM)은 최근 순차적 의사결정 과제를 수행하는 자율 에이전트로서의 가능성을 보여주었습니다. 그러나 대부분의 기존 방법은 계획 없이 탐욕적으로 행동하거나 환경 피드백에 적응하지 않는 고정된 계획에 의존합니다. 이로 인해 문제의 복잡성과 계획 수평선이 증가할수록 LLM 에이전트의 순차적 의사결정 성능이 저하됩니다.

우리는 환경 피드백에 따라 LLM 에이전트가 자체 생성한 계획을 적응적으로 수정할 수 있는 폐쇄형 접근 방식인 **AdaPlanner**를 제안합니다. AdaPlanner에서는 LLM 에이전트가 계획 내 수정과 계획 외 수정을 모두 활용하여 피드백을 기반으로 계획을 적응적으로 수정합니다.

환각(hallucination) 문제를 완화하기 위해 다양한 작업, 환경 및 에이전트 기능에서 계획 생성을 촉진하는 코드 스타일의 LLM 프롬프트 구조를 개발했습니다. 또한, 성공적인 계획을 몇 가지 예시로 활용하는 **기술 발견 메커니즘(skill discovery mechanism)**을 제안하여 에이전트가 더 적은 작업 시연으로 계획을 수립하고 수정할 수 있도록 지원합니다.

ALFWorld 및 MiniWoB++ 환경에서 수행한 실험 결과, AdaPlanner는 기존 최첨단 기준 성능을 각각 3.73% 및 4.11% 향상시켰으며, 샘플 사용량은 각각 2배 및 600배 더 적었습니다. AdaPlanner의 구현 코드는 [GitHub 저장소](https://github.com/haotiansun14/AdaPlanner)에서 확인할 수 있습니다.

## Introduction

Autonomous agent로의 LLM 방법론은 크게 두가지로 분류할 수 있다.

1. Open-loop system  
사전에 결정된 계획을 바탕으로 작업을 진행하지만, feedback 적용은 하지 않는다.  
간단하고, 계산 비용이 저렴하다.  
변화에 대한 적응이 부족하고, 비효율적인 계획을 생성할 수 있다.  
2. Closed-loop system  
environment 피드백을 통합하여 계획을 조정한다. 따라서 더 유연한 방식으로 작동할 수 있습니다  
계획을 고정하고, 피드백에 따라 실행 작업만 변경한다.  
greedy한 방법으로 단기적인 해결에만 집중하여 장기적인 측면에서는 부정적일 수 있다.  

![그림1](/assets/images/ADAP/1.png "그림1"){: .align-center}  

이러한 두가지 문제를 해결하기 위해 AdaPlanner를 제안한다.

planner와 refiner로 이뤄진 Closed-loop system

- planner: 작업을 관리 가능한 하위 목표로 분해하고 각 목표에 대한 환경 피드백 예측
- refiner: 실행 중 두 가지 유형의 환경 피드백 대응

피드백

- in-plan feedback: 예측된 결과와 일치하는 피드백. refiner는 LLM을 사용해 피드백에서 주요한 정보를 자연어로 추출한다. 이 동작은 ask_LLM()이라는 action으로 표현한다.
- out-of-plan feedback: 예측과 일치하지 않는 피드백. refiner는 전체 계획을 수정하고 중간 지점부터 현재 작업을 다시 수행한다.

AdaPlanner는 피드백 구조에 대한 사전 지식이 필요 없고, agent가 처음부터 시작할 필요 없이, 즉시 수정된 계획을 사용할 수 있게 한다.

Code-based prompting을 사용하여 정확한 계획 수립과 수정이 가능하고, hallucination을 완화한다.

Skill discovery process를 포함하여 성공한 기록을 저장하고, 사용할 수 있게 한다.

## Preliminary

![그림2](/assets/images/ADAP/2.png "그림2"){: .align-center}  

## AdaPlanner

planner는 포괄적인 계획을 생성한다. 

**초기 계획:** $$\rho_0(P_0|g,o_1)$$

P: plan, g: text ground task definition, o: environment observed

refiner는 피드백을 바탕으로 Plan을 수정한다.

**in-plan refinement:** $$\pi(a'_{>t}|g,c_{>t}\cup\{h_t\},P_0)$$

a: action, c: context, h: 시간 단계 𝑡에서 컨텍스트를 통해 획득

존재하는 Plan에서 더 나은 action을 선택하도록 유용한 정보를 넣어주는 일회성 동작

**out-of-plan refinement: $$\rho(P_t|g,c_t,P_{t-1})$$**

피드백을 전체 plan을 수정하는데 사용함

Skill memory는 이전에 성공한 plan과 상호작용을 저장한다.

### Plan Generation via Code-Based LLM Prompting

코드를 사용하는 것이 자연어보다 모호성과 상호작용에서의 실수를 줄여준다. 이를 바탕으로 LLM의 hallucination을 줄일 수 있다.

Pythonic code prompts를 사용해 프롬프트를 작성했다.

![그림3](/assets/images/ADAP/3.png "그림3"){: .align-center}  

Programming based planning example

![그림4](/assets/images/ADAP/4.png "그림4"){: .align-center}  

Initial Plan을 만들기 위해 task description과 사용가능한 action을 넣는다. 또 가능하다면 example을 넣는다.

Plan은 solution이라는 함수로 만들어지고, agent와 start_from을 파라미터로 갖는다.

sub-plan은 주석 형태의 설명과 동작부, 테스트하는 assert 구문으로 이뤄진다. 

### Adaptive Closed-Loop Plan Refinement

Initial Plan이 만들어지면 AdaPlanner는 LLM이 구문 오류를 고치도록 한다. 그 이후 코드가 동작하고, 결과를 가지고 수정을 한다.

**In-Plan Feedback and Refinement via ask_LLM() Action.**

ask_LLM()으로 동작하고 주로 특정 정보를 parsing하거나 추론이 필요한 경우 사용한다. 기존 code 기반 planning 방법론은 결과로 나오는 구조에 대한 정보를 사전에 알기 힘들어서 parsing이 어려운데, 이를 쉽게 해결한다. ask_LLM은 planner가 필요하다고 판단될 때마다 포함된다.

**Out-of-Plan Refinement with the Refine-Then-Resume Mechanism.** 

sub-plan이 실행된 이후에 assert를 통해서 현재 plan이 예상대로 잘 진행되고 있는지 확인한다. 여기서 오류가 발생하면, AdaPlanner는 계획의 수정을 진행한다. Plan을 수정하기 위해 agent에게 현재 실행 상황에 대한 정보를 얻고, 이 정보를 활용해 전체 Plan을 바꾼다. 또한 수정 전후의 계획을 비교해 시작 지점(start_from)을 다시 설정한다. 이 방법으로 처음부터 시작하는게 아니라 중간부터 시작해서 효율성을 높인다. 

### Skill Discovery

example을 확보하기 위한 방법

**Skill Acquisition**

처음 보는 task가 있을 때, 적은 수의 few-shot이나 zero-shot sample을 넣고  closed-loop planning을 반복하여 문제를 해결하고, 성공하면 이를 저장함

**Skill Filtering**

발견된 방법을 프롬프트에 통합했을 때와 아닐 때의 성능을 비교하여 성능향상이 있는 방법만 저장함.

## Evaluation

**Main Results**

ALFWorld

![그림5](/assets/images/ADAP/5.png "그림5"){: .align-center}  

MiniWoB (computer task)

![그림6](/assets/images/ADAP/6.png "그림6"){: .align-center}  

feedback이 제공되는 9개 task에서는 가장 좋은 성능을 보여 AdaPlanner가 feedback을 잘 받아들인다는 것을 보여준다. feedback이 없는 task도 좋은 성능을 보인다.

![그림7](/assets/images/ADAP/7.png "그림7"){: .align-center}  

적은 수의 샘플(예제)로도 높은 성능을 얻을 수 있다.

![그림8](/assets/images/ADAP/8.png "그림8"){: .align-center}  

(a). refinement 횟수에 따른 성능 변화. 예제의 수와 무관하게 피드백만으로도 성능이 향상된다.

(b). reflexion과 비교.

![그림9](/assets/images/ADAP/9.png "그림9"){: .align-center}  

다른 모델에 비해 적은 수의 반복으로 문제를 해결한다.

**Code Interface Mitigates Hallucination**

GPT-3.5-turbo가 가장 뛰어난 모델이나 GPT-3, GPT-3.5에 비해 여러 방법론에서 모두 성능이 떨어진다. 그 이유는 hallucination의 발생으로 보인다. 

그래도 AdaPlanner가 hallucination에 강건한데 이는 code-based 프롬프트를 사용해 LLM에게 더 형식적이고 제한된 생성 공간을 제공하기 때문으로 보인다.

비교를 위해 code based prompt를 사용하지 않는 AdaPlanner를 만들면 성능이 차이가 난다. (c)

![그림10](/assets/images/ADAP/10.png "그림10"){: .align-center}  

**Skill Discovery Improves Sample Efficiency**

![그림11](/assets/images/ADAP/11.png "그림11"){: .align-center}  

Long-term memory 기법을 활용하는 skill discovery는 성능 향상에 도움을 준다. 

## Conclusion

in-plan refinement와 out-of-plan refinement 방법론으로 주어진 환경 정보를 최대한 활용하고, code-based prompt로 hallucination을 완화했다.

AdaPlanner의 한계는 여전히 복잡한 작업을 해결하기 위해 여러 예제가 필요하다는 점이다.