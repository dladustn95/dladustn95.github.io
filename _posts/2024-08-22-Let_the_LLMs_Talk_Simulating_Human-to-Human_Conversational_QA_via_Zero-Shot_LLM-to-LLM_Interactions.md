---
title:  "Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions 논문 읽기"
excerpt: "Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions 논문 읽기"

mathjax: true
categories:
  - NLP
tags:
  - [DeepLearning, NLP, LLM, UserSimulation, WSDM24]
date: 2024-08-22T18:00:00+09:00
last_modified_at: 2024-08-22T18:00:00+09:00
---
  
# Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions

## Abstract

대화형 질문-응답(CQA) 시스템은 사용자와의 상호작용을 통해 효과적으로 정보를 검색할 수 있는 대화형 검색 시스템을 만드는 것을 목표로 합니다. 인간 간의 대화를 재현하기 위해 기존 연구에서는 인간 작업자들이 질문자(학생)와 응답자(교사)의 역할을 맡습니다. 이러한 방식이 효과적이긴 하지만, 사람이 하는 작업은 시간이 많이 소요되고 일관성이 없으며 확장성이 떨어진다는 문제가 있습니다. 이러한 문제를 해결하고 대화형 질문-응답 시뮬레이션에서 대형 언어 모델(LLM)의 적용 가능성을 조사하기 위해, 우리는 교사-학생 상호작용을 시뮬레이션하는 zero-shot learner LLM을 사용하는 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 두 개의 LLM이 특정 주제에 대해 상호작용하는 방식으로 구성되며, 첫 번째 LLM은 학생 역할을 맡아 주어진 검색 주제를 탐구하기 위한 질문을 생성합니다. 두 번째 LLM은 교사 역할을 맡아 질문에 답변하며, 주어진 주제에 대한 추가 정보(텍스트 포함)를 제공합니다. 우리는 GPT-4 모델을 제로샷 프롬프팅하여 학생과 교사 역할을 구현합니다. LLM이 CQA 상호작용을 시뮬레이션하는 데 있어 얼마나 효과적인지 평가하고, LLM이 생성한 대화와 인간이 생성한 대화 간의 차이를 이해하기 위해 다양한 관점에서 시뮬레이션된 데이터를 평가합니다. 먼저 자동 평가와 인간 평가를 통해 교사의 성과를 평가합니다. 다음으로, 학생의 성과를 평가하며, LLM이 생성한 질문과 인간이 생성한 질문 간의 차이를 분석하고 비교합니다. 또한, 두 데이터 세트에서 최신 독해 모델을 벤치마킹하여 LLM 성능을 철저히 검토하는 광범위한 분석을 수행합니다. 우리의 결과는 교사 역할을 맡은 LLM이 더 길고 정확하며 완전한 답변을 생성한다는 것을 보여줍니다. 학생 역할을 맡은 LLM은 더 다양한 질문을 생성하며, 주어진 주제의 더 많은 측면을 다룹니다.

## Introduction

Conversational Question-Answering (CQA) 시스템은 관련된 정보를 찾고, 사용자와의 상호작용을 통해 문제 해결을 위해 필요한 정보를 얻는 방법으로 좋은 성과를 거두었다. 기존 작업은 사전 정의된 주제에 대해서 작업자들이 질문자(student)와 답변자 (teacher) 역할을 반복적으로 수행하며 데이터를 모은다. 

그러나 이전 방법은 대화 데이터를 만들기 위해 대규모 작업자를 필요로 하고는 단점이 있다. 시간이 많이 걸리고, 리소스를 많이 필요로 하며, 비용이 많이 들 수 있다. 인간 작업자에게만 의존하면 생성된 대화의 품질과 일관성이 보장되지 않는다. 특히 인간 질문자 (student)는 자신의 배경 지식 범위 밖의 주제에 대해서 좋은 질문을 만들지 못한다. 

예를 들어 주제가 지리학인 경우 인간은 지리학에 대한 전문 지식이 없다면 관련 주제에 대해서 적절한 질문을 만들 수 없다. 반대로 LLM은 방대한 양의 배경지식을 활용해 지리학 전문가 역할을 효율적으로 할 수 있다. 따라서 본 논문에서는 대화 데이터를 생성하는 자동화된 방법을 제안하여 인간 작업자의 의존성을 줄이고, 작업을 효율적으로 만든다.

대화 관련 연구에서 user simulation에 대해 연구가 되고 있지만 기존 연구들의 경우 시스템 발화에 대해 답변을 하는 수동적인 형태를 띄고 있다. 그러나 real world에서 사용자의 행동은 능동적인 형태와 수동적인 형태가 섞여 있으며, 필요로 하는 정보에 대한 질문을 하는 형태로 대화를 시작하고 이끌어 나간다.

본 논문에서는 LLM이 능동적인 user simulator로써 효율성이 있는지 확인한다. QuAC에서 사용된 teacher-student 대화 시뮬레이션을 복제해서 사용한다. 여기서 LLM은 사람을 대체하고, 평가를 진행하여 LLM과 인간 작업자를 비교한다.

본 논문에서는 세개의 Research question에 대해 답변한다.

1. how can we employ LLMs to generate such simulated conversations effectively and automatically?
논문에서는 zero-shot LLM-to-LLM simulation framework를 사용했다. student LLM은 다양한 질문을 통해 주제를 탐구하는 목표를 갖고, teacher LLM은 질문에 대해 완전하고 정확한 답변을 주는 것이다. 논문은 두 LLM을 zero-shot GPT4를 사용해 구현했다.
2. how can we evaluate the role of LLMs in CQA simulation?
3. how do LLM- and human-generated conversations compare?

teacher LLM 성능 측정을 위해 같은 질문에 대한 LLM과 사람 각각의 답변을 비교한다.
student LLM 성능 측정을 위해 LLM과 사람의 패턴과 질문-답변 행동을 분석한다. 이때 LLM이 생성한 질문이 더 넓은 범위의 주제를 포함하는 것을 발견했다.
SOTA 기계독해 모델을 사용해 두가지 데이터셋의 성능을 비교한다.

성능을 측정한 결과 LLM에서 생성한 답변이 일반적으로 더 길고, 포괄적인 것을 확인했고, 사람보다 더 일관되고 유창한 답변을 생성한다. Human eval에서는 LLM teacher가 더 정확한 답변을 한다는 것을 확인했다. 또 reading comprehension 모델이 LLM이 만든 데이터셋에서 더 좋은 성능을 보이는 것을 발견했다. 이는 LLM이 생성한 대화가 특정한 bias를 갖게 되고, 그 안에서 일관성이 향상되기 때문이라고 한다.

## Methodology

### Problem Setting

논문에서는 student와 teacher가 상호작용을 하여 question-answering 대화를 하는 information-seeking conversation을 시뮬레이션한다. 그리고 **QuAc!** 데이터셋의 setting을 사용한다. 데이터는 위키피디아 article에 대한 토론이고, 두 명의 작업자가 각각 student / teacher 역할을 나누어 맡는다. teacher는 전체 문서에 접근할 수 있고, 학생의 질문에 대한 답변을 생성하는 것이 목표이다. student는 문서의 일부 내용만 보여지며, 이 정보를 바탕으로 관련 질문을 하고 주제에 대해 알아보는 것이 목표이다. 

### Task Formulation

대화는 Wikipedia article title $$t$$로 부터 시작된다. student에게는 제한된 정보만 주어지는데, section header $$h$$와 first paragraph $$b$$가 주어진다. teacher에게는 추가로 full text of section $$s$$가 주어진다. 

대화는 student의 시작 질문 $$q_0$$으로부터 시작되고, teacher는 그에 대한 답변 $$a_0$$을 준다. 이전 연구처럼 teacher는 자유롭게 답을 주는 것이 아닌 주어진 텍스트의 span을 선택해 그것을 바탕으로 답변을 제공한다.

### Model Framework Overview

![그림1](/assets/images/CQALLM/1.png "그림1"){: .align-center}  

Wikipedia page의 정보와 instruction을 사용해 student는 Question Generation component $$\phi_S$$에서 질문 $$q_0$$를 생성한다. 

생성된 질문은 Question Validation component에서 구조적 무결성을 검증한다. 이 과정에서 구조적 문제가 발견되면 뒤로 돌아가 질문을 다시 생성한다. 

이후 생성된 질문을 $$instruction_T$$와 함께 teacher로 전달한다. 답변 $$a_0$$ 을 생성하고, Answer Validation component에서 앞서 언급한 span이 포함되어 있는지를 체크한다. 

$$a_0$$가 적절하지 않다면 Prompt Selection Teacher component에서 적절한 prompt를 선택하고 다시 정답을 생성하게 한다.

적절한 정답을 생성하면 이를 student에게 전달하고, student는 prompt selection student component에서 다음 질문 $$q_i, i>0$$를 생성하기 위한 prompt를 선택한다. 그리고 Question Generation component에서 다시 질문을 생성한다. 

이러한 상호작용이 중단 기준이 적용될 때까지 반복된다. 

![그림2](/assets/images/CQALLM/2.png "그림2"){: .align-center}  

### Teacher Simulation

![그림3](/assets/images/CQALLM/3.png "그림3"){: .align-center}  

Answer Generation Component($$\phi_T$$)는 주어진 질문에 대해 정보 $$t, b, h, s$$를 사용해 Instruction을 만든다. 답변으로 s에서 적절한 text span을 찾아 복사해 사용하라는 instruction을 주었다. 답을 찾을 수 없을 때에는 ‘I cannot find the answer’라고 말하게 했다. 읽기 힘든 너무 긴 문장을 생성하는 것을 막기 위해, 선택한 span이 최대 40 토큰을 넘지 않아야 한다는 지시 사항과, 텍스트에서 가장 짧은 span을 선택하라는 지시사항을 추가했다.

Answer validation & regeneration Component($$\sigma_T$$)는 답변을 검증한다. valid answer는 
 1. s에서 얻은 text span이 포함되어 있거나, “I cannot find the answer”라는 답변이 있는지 
 2. 답변을 b가 아닌 s에서 복사한 것이 맞는지
를 확인한다. 
각각을 검증하기 위해 간단한 text search를 수행했다.
1번 검증에서 LLM이 s의 괄호 안 내용을 누락하거나,  띄어쓰기 여러개를 무시하는 특성이 있어 이를 normalize해서 확인했다. 여기서 오류가 발생하면 “Please copy the answer exactly from the given text”를 prompt에 추가했다. 

2번 검증에서 문제가 발생하면, “Please answer from the given section not the given background description” prompt를 추가했다.

검증이 끝난 답변은 student simulator에게 전달되고, 이 검증-생성 과정에 대한 threshold를 두어 이를 초과하여 반복하면 답을 찾을 수 없음으로 간주하고 반복을 끝낸다.

### Student Simulation

Question Generation Component($$\phi_S$$)는 질문을 생성하여 $$t, b, h$$로 주어진 정보에 대해 알아간다. 

Question validation ($$\sigma_S$$)는 질문의 구조를 검증한다.

제안한 framework에서 한번에 하나의 질문만 있음을 가정하지만, LLM이 여러개의 질문을 만들 때가 있다. 따라서 25단어를 초과하지 않고, 줄바꿈 문자 또는 1. 2. 와 같은 열거형 항목이 없음을 검증한다. 

Prompt selection for student ($$w_S$$) 

대화가 진행됨에 따라 질문이 주어진 정보와 관련이 있음에도 불구하고, full text of section $$s$$로 답변할 수 없는 경우가 생긴다. 이를 해결하기 위해 teacher가 주어진 질문에 답할 수 있는 능력을 지속적으로 체크하고, student prompt에 조정을 하여 질문의 퀄리티를 향상시키는 것이 중요하다.

“I cannot find the answer” 답변이 왔을 때 지나치게 구체적이라 주어진 $$s$$로 답할 수 없는 질문이 생길 때가 있다. 이를 해결하기 위해 아래의 4가지 프롬프트 중 하나를 선택하여 추가한다.

1. Ask a general question and do not ask a too specific question  
2. Ask a question starting with where, when, or who  
3. Ask a question about what is interesting in this article
4. Ask a question about another aspect of the topic

프롬프트를 추가하여 지나치게 구체적인 질문을 생성하는 것을 방지하고, user simulator에게 추가 단서와 정보를 제공한다. 이 방법은 user simulator가 주어진 정보를 보다 효율적으로 알아보게 만들어 궁극적으로 전반적인 이해도를 향상시킨다.

## Teacher Evaluation

### Experimental Setup

QuAC 데이터에서 50개 대화 셋 선택. 이 질문을 가지고 teacher LLM이 answer를 생성하게 하고, 원본 데이터 (사람이 만듬)와 비교

두 개의 답을 사람에게 correctness, completeness, naturalness 항목에 대해 평가하도록 하고, 둘 중 더 선호하는 시스템을 선택하도록 했다.

- correctness: 주어진 답변(text span)이 질문에 대한 올바른 답변인지 평가
- completeness: 답변이 완전하고, 포괄적인지 평가한다. 답변이 정확한 것과 완전한 것은 다른데, 예를 들어 어떤 가수의 앨범이 어떤 것이 있는지에 대한 질문이 있을 때, 더 완전한 답변은 하나가 아닌 더 많은 앨범에 대한 정보를 나열하는 것이다.
- naturalness: 답변의 유창성(fluency)과 사람다움(human-likeliness)을 평가. QuAC과 teacher simulator 모두 text span을 답변으로 사용하지만 QuAC의 데이터는 부자연스럽고, 완전한 문장 형태가 아닌 경우가 많았다.

![그림4](/assets/images/CQALLM/4.png "그림4"){: .align-center}  

### Experimental Results

50개의 대화 데이터에서 추출한 359개의 질문에 대해 평가를 진행했다.

77개의 질문에서 teacher simulator와 QuAC의 답변이 동일했고, 106개의 경우 둘 사이에 오버랩이 있어 둘 중 하나가 다른 하나에 포함되는 문장이었다. 나머지 176개의 경우 겹치지 않고 서로 다른 답변이었고, 특히 41개 질문에서 teacher simulator는 하나 이상의 답변을 했다.

![그림5](/assets/images/CQALLM/5.png "그림5"){: .align-center}  

![그림6](/assets/images/CQALLM/6.png "그림6"){: .align-center}  

사람이 평가한 결과 (Fleiss’ kappa = 0.4365)

모든 면에서 teacher simulator가 더 나은 평가를 받았고, 특히 대부분의 참여자가 teacher simulator를 더 선호하는 system으로 선택했다. QuAC 데이터는 불완전하거나 문법적으로 틀린 text span을 선택하는 경우가 많았는데 teacher simulator는 여러 text span을 선택할 수 있어 이런 현상을 개선할 수 있었다. 

더불어 주어진 text에서 정답을 선택하도록 제한하여 LLM의 hallucination을 줄였고, 검증 가능하게 만들었다. 이 결과는 crowd sourcing을 대체하는 LLM의 가능성을 보여주고, LLM이 충분한 정보를 가지고, 일련의 검증 작업이 병행되면 충분히 대화에 참여할 수 있는 가능성을 보여준다.

많은 작업자들의 코멘트에서 answer simulator가 더 사실적이고, 대화를 편안하게 한다는 것을 확인했다. 특히 answer simulator의 답변이 더 길더라도 이 편안함 때문에 더 선호한다는 의견이 있었다.

## Simulation Evaluation

### SimQuAC Dataset

앞서 설명한 framework를 활용해 만든 새로운 데이터셋 SimQuAC을 공개한다. QuAC의 train data에서 342개의 conversation을 랜덤으로 선택하고 여기서 겹치는 topic을 제외한 334개의 topic에 대해서 simulator를 사용해 데이터를 만들었다. SimQuAC는 평균 1.32개의 answer span을 포함한다.

![그림7](/assets/images/CQALLM/7.png "그림7"){: .align-center}  

### Student Evaluation

어떤 모델이 더 나은 질문을 하고 주제에 대해 탐구하는지 평가하는 metric을 만드는 것은 어렵기 때문에 여러 언어적인 측면에서 비교를 하는 것으로 student simulator와 사람을 비교하려 한다.

Question comparison

![그림8](/assets/images/CQALLM/8.png "그림8"){: .align-center}  

표는 같은 topic에 대한 QuAC와 SimQuAC의 질문을 비교한 것이다. GPT-4가 사람보다 더 구체적이고 긴 질문을 하는 것을 확인할 수 있다. 또한 사람은 4번의 질문만 하는 반면 GPT-4는 그 이후에도 질문을 계속하고 있다. 

Coverage

![그림9](/assets/images/CQALLM/9.png "그림9"){: .align-center}  

두 student가 얼마나 많이 topic에 대해 탐구했는지 알아보기 위해 s가 정답으로 사용된 범위를 측정한다. (정답은 s의 text span을 복사해서 사용하기 때문) 

그림 (a)를 보면 SimQuAC가 훨씬 더 많은 범위를 커버한다.

Conversation flow

대화에서 질문이 자연스러운지와 주제 전환이 자연스러운지 평가한다. 논문에서는 주어진 컨텐츠 s의 순서를 따라 대화의 질문이 이어지면 부자연스러운 것이라고 가정했다. 질문에 대한 답변의 text span이 시작되는 index를 순서대로 정렬했다. 예를 들어 A, B, C 순으로 질문을 했는데, 답변에 대한 index가 B, A, C 순이 될 수 있다. 

이 두개의 순서를 비교하기 위해 Kendall rank correlation coefficient (KRCC, 1이면 순서대로, -1이면 반대 순서)를 사용했다. 

그림 (b)는 SimQuAC가 덜 순서대로 질문하는 것을 보여준다. 어느 순서가 더 자연스러운지 정의할 수는 없지만, 사람과 LLM의 행동에는 차이가 있음을 보여준다. 또한 무작위로 질문하는 것이 더 어려운 dataset이 될 것이고, 이는 모델이 student의 bias를 학습하는 것을 방지한다. 

### Reading Comprehension Benchmarking

여러 discriminative and generative reading comprehension PLM을 사용해 teacher 모델을 평가했다.이 모델은 SQuAD 데이터셋으로 학습한 PLM이고 추가 학습 없이 QuAC와 SimQuAC 성능을 평가했다. QuAC와 SimQuAC의 conversation에서 최대 3개의 question만 사용했다. 

![그림10](/assets/images/CQALLM/10.png "그림10"){: .align-center}  

실험 결과 SimQuAC 데이터 셋이 더 좋은 성능을 보이고, 이건 LLM이 만든 데이터가 학습에 더 좋은 context를 갖고 있음을 증명한다. 특히 Exact Match의 성능이 낮은 이유는 SimQuAC가 QuAC보다 더 긴 span을 갖기 때문이다. 그리고 SimQuAC는 no answer인 질문이 많은데 PLM은 무조건 답을 만들기 때문에 EM 성능이 낮다. 

## Conclusions and Future Work

두 개의 GPT-4가 질문을 생성하는 student 답변을 생성하는 teacher 역할을 맡아 서로 상호작용하는 framework를 제안했다. 두 역할을 하는 LLM 각각의 성능을 평가하였고, 본 논문에서 제안한 LLM을 사용하는 방법이 가능성이 있음을 보였다.

그러나 GPT-4만이 teacher instruction(text span 만 답변으로 사용)을 따르는 문제가 있었다.

LLM은 이런 시뮬레이션 상황에서 다양한 bias를 가질 수 있기 때문에 이를 완화하는 방법을 개발해야 한다.

논문에서는 인간이 하는 상호작용을 모방하기 위해 prompt를 직접 작성하였지만, 이런 방법은 시간이 오래 걸리기 때문에 자동으로 프롬프트를 작성하는 전략을 고민하는 것도 좋겠다.