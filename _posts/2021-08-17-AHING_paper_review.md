---
title:  "All That's 'Human' Is Not Gold:Evaluating Human Evaluation of Generated Text 논문 읽기"
excerpt: "All That's 'Human' Is Not Gold:Evaluating Human Evaluation of Generated Text 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP]
date: 2021-08-17T18:00:00+09:00
last_modified_at: 2021-08-17T18:00:00+09:00
---

# All That's 'Human' Is Not Gold:Evaluating Human Evaluation of Generated Text

## 1. Introduction
NLG 시스템의 성능을 평가하는데 있어 사람이 쓴 텍스트는 성능의 upper bound 역할을 했습니다. 
또한 성능을 평가하기 위한 좋은 metric이 없었기 때문에 human eval에 의존해야 했습니다.  
평가자는 전반적인 품질, 자연스러움, 인간스러움과 같이 텍스트를 전체적으로 평가하도록 요청받거나, 텍스트의 문법이나 일관성 같은 특정 부분을 평가하게 됩니다. 
그러나 정확한 평가 기준은 평가자의 재량에 맡겨졌습니다.  
논문에서는 비전문가에게 NLG모델로 생성한 텍스트가 얼마나 사람 같은지 평가하게 해 기존의 평가 방법이 모델의 품질을 얼마나 측정할 수 있는지 실험했습니다. 
또한 자세한 지침을 주거나, 주석이 달린 예제, pair example을 주는 것으로 평가자를 훈련시키고 평가자의 정확성을 개선할 수 있는지 실험했습니다.

## 2. Current human evaluation
![그림1](/assets/images/AHING_figure1.png "그림1"){: .align-center}  
Task의 인터페이스 환경
### 2.1 The Task
먼저 훈련되지 않은 평가자가 사람과 기계가 생성한 문장을 구별할 수 있는지 실험했습니다. 
평가자에게 사람이 쓰거나 기계가 생성한 문장 5개를 랜덤하게 주고, 4가지 점수로 평가하게 했습니다.
4가지 점수는 다음과 같습니다. 
- Definitely human-written  
- Possibly human-written   
- Possibly machine-generated  
- Definitely machine-generated  

완전히 사람이 썼다고 평가한 경우 왜 그렇게 선택했는지, 그 외에 경우는 어떻게하면 더 사람이 쓴 것처럼 바꿀 수 있는지도 물어봤습니다. 

### 2.2 Data
3개의 도메인에 대해서 텍스트를 모았고, 사람이 작성한 것과 각 모델이 생성한 것 각각 50개씩 문장을 모았습니다. 
모델은 GPT2와 GPT3를 사용했습니다. 
100단어 이상의 텍스트만 사용했고, 텍스트의 문장 중 100단어에 도달한 문장까지만 사용했습니다. 
또한 문장을 생성할 때 GPT3의 three shot 세팅을 사용했습니다.  

**Stories**  
STORY 데이터로 reddit writingprompt 데이터 셋에서 once upon a time으로 시작되는 텍스트를 수집했습니다. 
문장을 생성할 때 Once upon a time이라는 문구에 condition되게 했습니다. 
Story에 대한 콘텐츠가 사실상 제한이 없고 창작 영역이기 때문에 가장 open된 도메인입니다. 
또한 품질에 대한 필터링 없이 소셜미디어에서 수집되었기 때문에 noisy한 데이터이기도 합니다.  

**News Articles**  
Newspaper3k라는 크롤링 툴을 사용해 15개의 신문사에서 최근 지역 뉴스를 모았습니다. 
또한 문장을 생성할 때 기사의 헤드라인과 첫번째 문장을 주었습니다. 
뉴스 기사의 제목과 첫 문장이 종종 그 내용을 요약하기 때문에 생성된 내용은 주제를 따라야 합니다. 
조금 더 close한 도메인입니다. 
또한 최신 뉴스 데이터를 사용하여 모델이 학습 데이터에서 본 적이 없는 데이터를 사용했습니다.  

**Recipes**   
RecipeNLG 데이터셋에서 레시피를 수집했습니다. 
Prompt로 요리 제목과 재료 목록을 사용했습니다. 
주어진 재료를 사용해 제목에 설명된 요리를 만들어야 하기 때문에 가장 close한 도메인입니다. 
레시피는 명확한 명령으로 작성되기 때문에 예상치 못한 텍스트가 나올 자리가 없습니다.  

### 2.3 Participants
Amazon Mechanical Turk (AMT)를 통해 평가를 진행했습니다. 
6개의 task에 대해 각각 다른 130명의 평가자를 모집했습니다. 
평가자는 미국에 살고 좋은 신뢰도를 가진 평가자만 평가할 수 있게 했습니다. 

### 2.4 Result
![그림2](/assets/images/AHING_figure2.png "그림2"){: .align-center}  
GPT2는 평균 58%정도로 사람이 쓴 것과 기계가 생성한 텍스트를 그나마 잘 구분한다 할 수 있지만 gpt3는 50%로 구분하지 못했습니다. 
특히 story는 62에서 48로 큰 하락이 있었습니다. 
F1과 precision, recall도 다 하락했습니다. 
krippendorff’s. A는 Kappa score와 비슷하게 신뢰도를 측정하는 metric 입니다. 
1로 갈수록 신뢰할 만한데 아주 낮은 것을 볼 수 있습니다. 
평가자들마다 같은 텍스트를 보고도 다 다르게 평가했습니다.
GPT3로 생성한 텍스트의 경우 human이라고 평가한 비율이 올라갔습니다.

### 2.5 Analysis
논문에서는 평가하기 어려워진게 모델의 발전을 의미하기도 하지만 현재 평가 방법의 한계도 지적합니다. 
평가자가 어떤 부분에 중점을 두어 평가했는지 알아보기 위해 GPT3와 사람이 쓴 것을 구분한 150개의 응답을 골라 3가지 카테고리로 분류했습니다.  
- Form: 문법 포맷, 스타일 등 
- Content: 텍스트의 의미
- machine capabilities: 기계가 생성가능한 언어에 대한 사람들의 인식 (기계는 감정을 표현 못할 것이다.)  

Form에 대한 코멘트가 content에 대한 코멘트보다 두배 더 많았는데, gpt3는 충분히 말을 잘하기 때문에 form을 사용하는 것은 인간과 기계를 구분하는데 도움이 되지 않습니다. 
평가자들이 답변에 대해 제시한 이유들이 종종 서로 모순된다는 것을 발견했습니다. 
대부분의 평가자들이 NLG 모델을 과소 평가하거나 잘못 이해한다는 점을 발견했습니다. (AI가 literally라는 표현을 못쓸것이라 생각하거나 사랑에 대한 말을 못한다고 생각하는 등)

## 3.  Can we train evaluators to better identify machine-generated text?
기존의 평가 환경이 가지는 짧고 비용이 적게 드는 특성을 유지하면서 사람들의 평가 능력을 향상시키는 세가지 방법을 테스트했다고 합니다. 
GPT3에 대해서만 평가를 했고 3개의 도메인에 3개의 학습 방법을 사용했습니다.

### 3.1 Evaluator Training Methods
**Training with Instructions**   
![그림3](/assets/images/AHING_figure3.png "그림3"){: .align-center}  
평가자들에게 텍스트 중 어느 부분을 주의 깊게 봐야할 지 알려줍니다. 
기계가 생성한 텍스트의 어느 측면을 봐야 구분하기 좋은지, 또 평가자가 오해할 수 있는 부분을 알려줍니다.  

**Training with Examples**   
평가하기 이전에 3번의 연습과정이 주어집니다. 
텍스트의 어느 측면이 정답을 암시하는지 간단한 설명이 주어집니다. 
평가자가 풀은 뒤 정답을 알려줍니다.  
  
**Training with Comparison**  
사람이 쓴 것과 기계가 생성한 것 둘 다 동시에 보여주고 기계가 생성한 것을 고르게 합니다. 
Example training과 마찬가지로 정답을 알려주고 설명도 같이 알려줍니다.

### 3.2 Result
![그림4](/assets/images/AHING_figure4.png "그림4"){: .align-center}  
3가지 방법 모두 accuracy를 올리긴 했고 example training은 꽤 많이 올랐습니다. 
특히 story 도메인에서 많이 향상되는데 example training이 기계가 창의적인 텍스트를 생성하지 못한다는 편견을 깨는데 도움이 되기 때문이라고 합니다. 
또한 example과 comparison의 human이라고 한 비율이 낮아지면서 더 많은 텍스트를 기계가 한 것으로 예측했습니다. 
그러나 krippendorff’s A가 여전히 낮았습니다. 

### 3.3 Analysis
![그림5](/assets/images/AHING_figure5.png "그림5"){: .align-center}  
앞선 분석과 마찬가지로 평가자가 어떤 부분에 중점을 두어 평가했는지 조사했습니다. 
Example training의 경우 form과 machine capabilities의 비율이 줄고 content에 대한 비율이 늘어 앞서 설명했던 내용적인 측면에 중점을 두어야 한다는 점에 부합했습니다.

## 4. Discussion
![그림6](/assets/images/AHING_figure6.png "그림6"){: .align-center}  
다음 그림은 GPT3가 생성한 텍스트에 대해 평가자들이 각각 사람과 기계가 쓴 텍스트라고 평가한 이유입니다.
평가자들은 각각 다른 특징에 집중해 텍스트를 평가합니다. 
또한 텍스트의 같은 특징을 보고도 다른 결론을 내리기도 합니다.  

평가자는 기계의 텍스트 생성 능력을 과소평가하는 경향이 있었습니다. 
Example training을 받은 평가자는 기계가 생성한 텍스트에 대한 기대치가 더 높았고 
사람과 기계를 분류할 때 텍스트의 내용에 더 집중했음에도 불구하고 구분 능력의 향상이 크지는 않았습니다.  

평가자들은 인간의 속성이나 의도를 반영하는 텍스트를 기계가 생성할 수 없다고 생각합니다. 
그러나 현재 NLG 모델은 적어도 인간 속성이나 의도에 대한 텍스트를 생성할 수 있습니다. 
논문에서는 이에 대한 해결책으로 평가자 교육을 집중하는 것을 제안합니다. 
Example training에서 특별히 이것에 초점을 맞추지 않았음에도 machine capabilities가 감소하면서 정확도가 증가했음을 보였기 때문이라고 합니다. 

## 5. Conclusion
논문에서는 평가자들이 문법과 같은 텍스트의 표면 수준에만 초점을 맞추고, NLG 모델을 과소평가했음을 발견했습니다. 
또한 지금과 같은 방법으로는 사람과 기계가 생성한 텍스트를 구분하지 못하기 때문에 평가 방식을 바꿔야한다고 주장합니다. 
