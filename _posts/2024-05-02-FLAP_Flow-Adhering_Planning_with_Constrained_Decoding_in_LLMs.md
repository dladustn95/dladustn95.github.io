---
title:  "FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs 논문 읽기"
excerpt: "FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, ActionPlanning, NAACL24]
date: 2024-05-02T18:00:00+09:00
last_modified_at: 2024-05-02T18:00:00+09:00
---
  
# FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs

## Abstract

TODs(Task Oriented Dialogs)에서 에이전트의 계획은 중요한 작업입니다. 인간 에이전트들은 일반적으로 사용자 문제를 해결하기 위해 사전 정의된 워크플로우를 따라가며 작업을 분해하고 행동 가능한 항목으로 변환하며, 액션을 수행하기 위해 API를 실행합니다. 이 모든 것은 추론과 계획이 필요합니다. 최근 LLMs의 발전으로 인해 작업 계획 및 API 사용에 대한 이를 사용하는 시도가 증가했습니다. 그러나 LLMs의 계획의 충실성은 사전 정의된 워크플로우와 API 종속성에 대한 보장이 되지 않습니다. 또한 현실에서의 워크플로우는 종종 사용자 정의되어 있으며 변경될 수 있으므로 적응이 바람직합니다. 이를 연구하기 위해, 사전 정의된 플로우를 따르고 API 종속성을 보존하여 사용자 의도를 해결해야 하는 TODs의 충실한 계획 문제를 제안합니다. 이 문제를 해결하기 위해 우리는 제한된 디코딩과 미리보기 휴리스틱을 기반으로 한 플로우 준수 계획 알고리즘인 FLAP을 제안합니다. 우리의 알고리즘은 도메인 특정(계획/종속성) 데이터를 사용하여 LLMs를 파인튜닝할 필요를 줄이고, 사전 정의된 플로우에 빠르게 적응하도록 가능하게 하며, 다른 디코딩 및 프롬프트 기반의 기준선을 능가합니다. 더 나아가, 우리의 알고리즘은 더 작은 LLMs(≈ 7B)가 더 큰 LLMs(≈ 30B-40B)와 동등한 성능을 발휘할 수 있도록 합니다.

## Introduction

![그림1](/assets/images/FLAP/1.png "그림1"){: .align-center}  

## Task Formulation

### TOD Agent Scenario

Domain ($$D$$)

Intent ($$I$$)
Agent는 각 domain에 포함되는 intent를 처리할 수 있어야한다. 그러나 다른 도메인에 포함되는 intent는 처리하지 못해도 된다. 여행 domain의 agent면 계좌 개설과 같은 intent는 못해도 됨

Flows ($$F$$) : 
자연어로 표현된 연속된 지시 사항. ex) “checking out 전에 add-ons를 추천하라.” Agent는 intent를 해결하기 위해 step-by-step으로 이 flow를 따라갈 것으로 기대된다. Flow는 변경될 수 있다. Intent를 해결하기 위해 step으로 이뤄진 지시 사항.

APIs ($$A$$):
Agent는 internal API를 사용하여 시스템과 소통할 수 있다. 또한 유저의 입력을 바탕으로 동작을 실행한다. ex) 유저가 시카고에서 차를 예약하려 한다면 agent는 다음과 같이 API를 실행할 수 있다.
FindRentalCar(location=“Chicago”)

### Faithful Planning for Resolving Intent

현실 세계에서 user가 intent를 표현하면 agent는 APIs의 sequence로 이를 처리할 수 있어야한다. 이는 아래와 같은 이유로 agent의 reasoning과 planning 능력에 크게 좌우된다.

Identifying and following NL flows:
$F$에 대한 실행 단계는 주어진 $F$를 따라야 한다. 그러기 위해 agent가 $F$의 정확한 의미를 이해하고 이를 따라야 한다.

Task decomposition and dependency resolution: 
APIs는 다른 APIs에 의존적일 수 있고, 여러 APIs를 사용할 필요가 있다. FindFlight(airport=“ORD”) 이전에 GetAirports(city=“Chicago”) 호출해야 함. 즉 FindFlight(airport=GetAirports(city=“Chicago”))와 같은 형태. 따라서 agent는 흐름을 분석하고, APIs 사이의 종속성을 유지해야 한다.

이러한 문제를 해결하기 위해 본 논문에서는 새로운 방법을 제안한다. Domain, set of Flows, set of APIs (with parameter descriptions), user query $F$가 주어지면 이를 해결하기 위한 plan $P$를 생성한다. Plan은 $F$의 흐름 단계를 준수하고 $F$의 API 종속성을 유지하면서 사용자 쿼리를 이행하는 API 시퀀스로 구성된다.

실제 세상에서는 agent가 상황에 따라 계획을 동적으로 업데이트해야할 수도 있다. (dynamic planning, 사용자가 intent를 변경할 때 등이 있음) 그러나 본 논문에서는 이는 고려하지 않고 user가 의도를 변경하지 않는 static planning만 고려한다.

### Zero-shot Planning with LLMs

LLM의 zero-shot planning 성능을 확인해야 한다. LLM에 APIs의 정의와 입출력 parameter 정보, 자연어로 된 Flows, user query를 instruct로 사용했다.

All vs. relevant flow in-context:
Prompt에 모든 Flows를 주거나, user query에 관련한 Flow만 주는 실험을 진행했다. 전자는 전체 Flows에서 관련된 flow를 찾고, 이에 따른 planning을 해야하기 때문에 더 복잡한 문제가 된다.

Reasoning and acting:
최근 연구는 step-by-step thinking and acting이 목표 달성에 도움이 되는 것을 확인했다. 이것을 반영해서 thought를 생성하고, planning을 위한 APIs를 생성하도록 했다.

### Evaluation Metrics

Number of edits to fix the plan:
생성된 plan에는 gold plan과 동일한 step과 APIs만 포함되어야 한다. 따라서 생성된 것을 몇번 수정해야 gold plan과 동일해지는 지를 계산한다. 

Inconsistent steps/APIs:
생성된 plan의 step과 APIs가 description으로 주어진 dependency를 잘 따르고 있는지 계산한다.

Fine-grained metrics:
그 외에 API hallucination, API repetition, parsability 를 평가한다.

## Data Collection

### Domain Creation

다른 dataset에는 본 논문에서 제안하는 방법에 대한 데이터가 존재하지 않는다. Flows, APIs에 대한 description이 필요함. APIs의 dependency가 있어야하고, Flows의 각 단계를 실행하는데 필요한 APIs가 주어지지 않았다. 
본 논문에서는 4 domain, 13 flows, 64 APIs로 이뤄진 dataset을 만들었다. 

![그림2](/assets/images/FLAP/2.png "그림2"){: .align-center}  

### Test Query Generation.

Falcon-40B-instruct LLM을 사용해 intent와 관련한 user query를 생성하도록 했다. 생성된 query를 직접 선별하여 intent와 관련 없는 query는 제거했다. 생성된 query의 94%가 정확했고, 이 방법으로 intent당 20개씩 query를 생성하여 260개를 사용했다.

## Constrained Decoding for Planning

### Zero-shot Planning with LLMs ablation

![그림3](/assets/images/FLAP/3.png "그림3"){: .align-center}  

다양한 모델을 사용해 실험을 진행했다.

Effect of thoughts:
거의 모든 LLM이 thought를 생성하는 것이 Flow faithfulness 성능 향상에 도움이 되었다. 그러나 API repetition, hallucination이 증가하였고, plan parsability가 하락하였다. 몇몇 예시에서는 thought가 반복되는 현상이 나타났다.

Effect of parameter size:
LLM의 크기가 커질 수록 성능이 향상되었다. 또한 비슷한 크기라도 성능에 차이가 있었다.

Effect of pretraining:
ToolAlpaca와 같이 tool에 대해 pre-training된 모델은 hallucination이 높다. 이는 tool과 관련한 pre-train data에 대한 bias가 생기는 것으로 보인다.

All flows vs. relevant flow in prompt:
관련된 flow만 주었을 때 성능이 향상되는 것으로 보아, LLM이 여러 flow에 대해 혼란을 가진다는 것을 알 수 있다.

### Proposed Algorithm - Flow Adhering Planning algorithm

![그림4](/assets/images/FLAP/4.png "그림4"){: .align-center}  

FLAP에서는 flow steps과 APIs dependency를 그래프로 표현하여 디코딩에 사용한다.

### Constructing Dependency Graphs

API dependency graph는 APIs 파라미터의 input, output을 사용해 만들 수 있다. 

FindFlight(inputs:**airport**; outputs:flights)

GetAirport(inputs:city; outputs:**airport**)

여기서 GetAirport는 FindFlight의 부모가 된다.

Flow step에서도 이와 동일하게 graph를 만들 수 있다. 

planning을 생성하는 단계에서 이 graph를 참고하여 실행할 수 있는 APIs / Step을 관리할 수 있다.

### Constrained Decoding of Plans with Lookahead Heuristic

Text 생성 문제는 아래 수식과 같이 표현할 수 있다. 여기서 Y는 모든 생성 가능한 값들이다.

$$
y_t = argmax_{y'_t \in Y}P(y'_t|x)
$$

constrained decoding에서는 $P(y'_t|x)$를 아래와 같이 변환할 수 있다.

$$
P'(y'_t|x) = P(y'_t|x) + H(y'_t|x)
$$

여기서 $H(y'_t|x)$는 생성된 $y'_t$에 대한 제약 조건의 만족도 점수이다. 그러나 다음 토큰만 보는 것으로는 이 점수를 구하기 어려울 수 있다. lookahead mechanism으로 일정 길이 만큼의 미래의 생성 값을 보고 점수를 계산할 수 있다.

$$
H(y'_t|x) = H(y'_t, ..., y'_{t+L}|x)
$$

점수는 다음과 같이 lookahead 길이 L 만큼 생성하고 구하는 것으로 바꿀 수 있다.

Constrained plan generation을 위해 논문에서는 thougt를 생성하고 그에 해당하는 api를 생성하도록 하였다. 생성 과정에서 다음 토큰에 대해 하나의 thoght+API가 생성되는 것을 lookahead 길이로 삼고 candidate를 생성한다. 그후 제약 조건 만족도에 따라 candidate를 평가한다. 이 논문에서 제약 조건의 만족도 점수에 thought, api를 고려한다. thought, api는 이미 생성된 토큰에서 가져올 수도 있다. 

각가의 thought, api는 아래의 조건으로 평가한다.

**Generated thought to permitted flow step alignment ($H_{th:step}$)**

논문에서 thought는 다음에 실행될 step을 설명하는 문장이다. 따라서 생성된 thought가 주어진 flow step과 일치해야한다. 이를 계산하기 위해 다음의 수식을 사용한다.

![그림5](/assets/images/FLAP/5.png "그림5"){: .align-center}  

여기서 S는 Domain에서 모든 가능한 step의 종류이다. 

가중치는 a>b로 설정하면 모델이 flow를 이탈하는 것을 방지할 수 있다. 하나의 step이 여러 api를 호출할 수 있기 때문에 c>a로 설정하면 다음 step으로 이동하기 전에 현재 step을 마무리할 수 있게 한다.

ex) c=1 > a=0.5 > b=0.1

**Generated API to permitted APIs alignment ($H_{api:\bar {api}}$)**

![그림6](/assets/images/FLAP/6.png "그림6"){: .align-center}  

여기서 $\bar a$는 생성된 api이다. $A_p$는 실행 가능한 API, $A_c$는 이미 실행된 API이다. hallucination으로 생성된 API나 실행 불가능한 API의 경우 0 이상 1 미만의 값을 줘서 제약을 준다. 이 값은 tuning할 수 있게 했다. ex) 0.1

**Generated thought to user intent alignment ($H_{th:in}$)**

plan이 user query에 존재하는 intent와 일치해야 하기 때문에 이 값을 계산한다. $in$은 user query

$$
H_{th:in} = sim(th, in)
$$

**Generated thought to generated API alignment ($H_{th:\bar {api}}$)**

API 사용을 요구하는 thought를 생성해야 한다. 따라서 생성된 api와 thought의 유사도를 측정한다.

$$
H_{th:\bar {api}} = sim(th, \bar a_d)
$$

여기서 $\bar a_d$는 생성된 API의 description, 생성된 API가 hallucination이면 API 이름 그대로 사용

**Structural constraint ($F_{st}$)**

LLM이 미리 정해준 output 형식에 맞게 응답을 생성하였는지 평가한다.

$$
H'_c = a \times H_{th:step} + b \times H_{api:\bar {api}}+ c \times H_{th:in}+ d \times H_{th:\bar {api}}
$$

![그림7](/assets/images/FLAP/7.png "그림7"){: .align-center}  

이 값을 최종 제약 조건 점수로 사용한다.

$$
S(y'_t|x) = (1-\lambda) \times P(y'_t|x) + \lambda \times H_c^{y'_t}
$$

top-k beam을 사용해서 p(y’_t)를 계산한다.

k = 10, 람다=0.7로 사용

# Experimental Evaluation

### Experimental Setting

Mistral-7b-instruct, Mpt-7b-instruct를 사용.

structure를 유지하기 위해 2개의 예시를 넣고, **$F_{st}$**를 사용해 강제한다.

유사도 계산을 위해 pre-trained sentence transformer를 사용했다.

### Experimental Results

![그림8](/assets/images/FLAP/8.png "그림8"){: .align-center}  

Greedy search에 비해 beam search나 nucleus sampling이 성능 향상을 주지 못한다. 더 넓은 검색 공간을 탐색하고, 다양성을 주는 것이 도움이 되지 않음. 

Mpt 모델의 경우 FLAP을 사용하면 큰 차이의 성능 향상이 있고, Mistral의 경우 inconsistent steps 성능이 하락하지만 다른 모든 지표에서는 성능이 향상된다. 

Mistral에서 FLAP 없이도 좋은 성능을 보이지만 여러 APIs 호출이 필요한 step을 잘 해석하지 못한다. average API count per plan이 4.8로 gold인 6.84보다 낮다. 또한 inconsistent steps도 3.1%로 낮지만  inconsistent API는 40.3%로 높다. FLAP을 쓰면 복잡한 step을 잘 해석할 수 있고, inconsistent steps이 살짝 증가하지만, inconsistent API가 3~7%로 현저히 감소한다.

![그림9](/assets/images/FLAP/9.png "그림9"){: .align-center}  

Mistral의 예시

제약 조건 점수를 구성하는 요소를 제외하여서 ablation 실험을 진행한다. 
각각의 component가 빠지면 그와 연관된 성능이 하락하는데, mpt에서 step(**$H_{th:step}$)**을 제외하면 inconsistent step %가 늘어나고, intent(**$H_{th:in}$)**가 빠지면 Avg # of edit for %가 늘어난다. 
API($H_{th:\bar {api}}$)는 thought와 API를 일관성 있게 생성되게 한다. 이게 빠지면 최종적인 API가 맞더라도 정확하지 않은 thought가 생성된다.
그러나 Mistral에서는 그 효과가 불분명한데, Mistral이 이미 좋은 성능을 보이기 때문으로 생각된다.

![그림10](/assets/images/FLAP/10.png "그림10"){: .align-center}  

![그림11](/assets/images/FLAP/11.png "그림11"){: .align-center}  

FLAP을 사용한 7B  모델과 다른 LLM을 비교한 것인데 7B  모델이 더 좋은 성능을 보인다.

# Conclusions

사전에 정의된 workflow와 API dependency를 지키는 planning 모델을 만들기 위해 Heuristic한 lookahead 디코딩 알고리즘을 사용하는 FLAP 모델을 제안했다.

### Limitation

Simplifying Assumptions: 
사용자의 발화가 주어진 APIs와 Flow로 모두 처리 가능하다는 가정 하에 문제를 풀었다. “cannot help”라는 Flow를 추가하는 것으로 OOD 문제를 간단하게 해결할 수는 있다.
또한 데이터 셋에는 여러 계획이 가능한 경우도 포함되어 있지는 않다.

Static Planning:
이 논문에서는 유저의 요구가 변화하지 않는 static planning에 대해서만 다루고 있다. 그러나 real world에서는 사용자의 요구에 따라 agent가 plan을 변화시킬 수 있어야 한다. 
또한 API의 선택에만 집중하고 있고, API의 파라미터의 정확도는 크게 고려하지 않았는데, 이는 주로 dynamic planning에서 고려될 것이기 때문에 future work 로 남겨두었다.

Runtime:
Runtime이 길다는 단점이 있는데, 이는 구현이 비효율적으로 되어 있기도 하고, 컴퓨팅 자원의 한계이기도 하다.
이러한 구현을 효율화 하거나, 디코딩 알고리즘을 효율화 하는 방법(예를 들어 n개의 token마다 lookahead를 한다)을 연구할 수 있다.

Usage of Open Source LLMs:
제안한 방법이 logits 값을 필요로 하고, 실험 중에 모델이 바뀔 가능성이 존재하고, 비용이 들기 때문에 open source만 사용했다. 또한 자원의 한계로 40B 모델까지만 실험해볼 수 있었다.
