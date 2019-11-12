---
title:  "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"
excerpt: "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - NLP, RelationPrediction, LinkPrediction, ConvE
last_modified_at: 2019-11-11T11:11:00+09:00
---

## 서론
&nbsp;&nbsp;지식그래프(Knowledge graphs)는 지식베이스(Knowledge bases)에서 관계(relation)를 edge로 개체(entity)를 node로 바꾸어 만든 것이다. 
지식그래프는 search, analytics, recommendation, data integration에서 중요한 자원이다. 
그러나 그래프의 link가 누락되어 있는 불완전성으로 인해 사용에 어려움이 있다. 
예를 들어 DBpedia의 Freebase는 66%의 사람 개체의 출생지가 연결되어 있지 않다. 
이러한 link의 누락을 식별하는 것을 link prediction이라 부른다. 
지식그래프는 수백만개의 사실을 포함할 수 있다. 
따라서 link predictor는 파라미터의 수와 컴퓨팅 비용을 현실적으로 관리하는 방식으로 확장해야한다.  
&nbsp;&nbsp;이러한 문제를 해결하기 위해 link prediction 모델은 주로 제한된 수의 매개변수를 사용하고, 임베딩 공간에서의 행렬 곱같은 간단한 연산으로 구성되었다. 
DistMult는 parameter당 하나의 feature를 생성하는 embedding parameter간의 3방향 상호작용이 특징이다. 
빠르고 간단한 얕은 모델을 사용해 덜 표현적인 feature를 배우는 대신 큰 지식그래프로 확장할 수 있다.  
&nbsp;&nbsp;얕은 모델의 feature 수를 늘리는 방법(표현력을 높이는 방법)은 embedding 사이즈를 증가시키는 것뿐이다. 
그러나 이러한 방법은 더 큰 지식그래프로 확장되지는 않는데, embedding parameter의 수는 그래프의 개체와 관계 수에 비례하기 때문이다. 
예를 들어 DistMult와 같은 얕은 모델의 embedding 사이즈가 200이라 가정하고 Freebase에 적용하면 이것의 parameter 용량은 33GB이다. 
embedding 사이즈와 관계 없이 feature의 수를 늘리려면 feature의 여러 레이어를 사용해야한다. 
그러나 feature가 완전연결 레이어로 이뤄진 이전의 다층 지식그래프 임베딩구조는 overfit되기 쉬웠다.
얕은 구조의 크기 조절 문제와 완전연결된 깊은 구조의 overfit문제를 해결하기 위해서는 parameter를 효율적으로 사용하고, Deep network로 구성된 빠른 연산자가 필요하다.  
&nbsp;&nbsp;컴퓨터비전에서 주로 사용되는 convolution 연산자는 위의 속성을 만족한다. 
게다가 다층 convolution 신경망을 학습할때 overfitting을 제어할 수 있는 강력한 방법론이 존재한다.  
&nbsp;&nbsp;이 논문에서는 지식그래프에서 embedding위에서 2D convolution 연산을 사용해 지식그래프의 손실된 link를 예측하는 모델, ConvE를 소개한다. 
ConvE는 단일 convolution layer, embedding dimension에 대한 projection layer, 내부 product layer로 구성된 간단한 다층 convolution 구조이다.  
+ 간단하지만 경쟁력있는 2D convolutional link prediction 모델 ConvE를 소개한다. 
+ 3배 빠른 훈련과 300배 빠른 평가를 가능하게 하는 1-N scoring 기법 개발.
+ parameter 효율적인 모델을 만듬, DistMult와 R-GCNs에 비해 각각 8배, 17배 적은 parameter로 FB15k-237에서 더 나은 점수를 달성.
+ ConvE와 다른 얕은 모델간의 성능 차이가 indegre와 PageRank로 측정한 지식그래프의 복잡도에 비례하여 증가한다.
+ 주로 사용되는 link prediction dataset의 inverse relation으로 인한 평가의 어려움을 조사하였고 inverse relation을 제거한 버전의 dataset을 소개하여 간단한 규칙 기반 모델로 해결할 수 없도록 함.
+ ConvE와 이전에 제안된 모델을 위에서 소개한 dataset으로 평가한 결과 MRR에서 state-of-the-art를 달성    

## 관련 연구
&nbsp;&nbsp;여러 link prediction 모델이 제안되었다. Translating Embedding 모델(TransE), Bilinear Diagonal 모델(DistMult)등. 
본 논문의 모델과 가장 유사한 모델은 Holographic Embedding 모델(HolE)이다. HolE는 cross-correlation을 사용한다. 
그러나 HolE는 비선형 feature의 여러 layer를 학습하지 않아서 이론적으로 ConvE보다 표현력이 떨어진다.
&nbsp;&nbsp;ConvE는 link prediction에서 2D convolutional layer를 사용한 최초의 모델이다. 
*Graph Convolutional Networks*(GCNs)는 convolution 연산자가 그래프의 locality 정보를 생성하기 위해 사용되었다. 
그러나 GCN framework는 undirected 그래프에 한정되어 있고, 지식그래프는 directed 그래프이다.  
&nbsp;&nbsp;여러 convolutional 모델들이 여러 자연어처리 task 해결을 위해 사용되고 있다. 
그러나 대부분의 연구들은 embedding 공간 안에서 단어의 순서와 같이 embedding의 시간적 순서에 작동하는 1D convolution을 사용한다. 
이 논문에는 embedding에 대해 공간적으로 작용하는 2D convolution을 사용한다.    

### Number of Interactions for 1D vs 2D Convolutions
 1D convolution 대신 2D convolution을 사용하면 embedding 사이의 추가 상호 작용 지점을 통해 모델의 표현성이 향상된다. 
예를 들어 두개의 1D embedding을 concatenate 한다고 생각하자.  
  $$(\begin{bmatrix}a & a & a \end{bmatrix};\begin{bmatrix}b & b & b \end{bmatrix}) = \begin{bmatrix}a & a & a & b & b & b\end{bmatrix}.$$  
사이즈가 3인 필터를 사용하면 concatenation point에서 두 embedding의 상호작용을 모델링 할 수 있다.  
&nbsp;&nbsp;두개의 2D embedding을 concatenate 한다고 가정하면 다음과 같은 결과를 얻는다.
  $$(\begin{bmatrix}a & a & a\\ a & a & a \end{bmatrix};\begin{bmatrix}b & b & b\\ b & b & b \end{bmatrix}) = \begin{bmatrix} a & a & a \\ a & a & a \\ b & b & b \\ b & b & b \end{bmatrix}.$$  
사이즈가 3x3인 필터를 사용하면 concatenation line에서 두 embedding의 상호작용을 모델링 할 수 있다.  
&nbsp;&nbsp;다음과 같은 다른 패턴도 사용할 수 있다.  
  $$\begin{bmatrix}a & a & a \\ b & b & b\\ a & a & a  \\ b & b & b\end{bmatrix}.$$  
이 경우에는 2D convolution 연산자가 a, b 사이의 더 많은 상호작용을 모델링 할 수 있다. 
따라서 2D convolution이 1D convolution 보다 더 많은 feature 상호작용을 추출할 수 있다.    

## Background
&nbsp;&nbsp;*knowledge graph* $$\mathcal{G} = \left\{ \left(s,r,o\right) \right\}\subseteq\mathcal{E}\times\mathcal{R}\times\mathcal{E}$$는 관계 $$r\in\mathcal{R}$$와 트리플의 주어와 목적어로 표현되는 두 개체 $$s,o\in\mathcal{E}$$를 포함하는 트리플의 집합으로 공식화할 수 있다. 
각각의 트리플 $$\left(s,r,o\right)$$은 개체 $$s$$,$$o$$ 사이의 관계 $$r$$을 나타낸다.  
&nbsp;&nbsp;*link prediction*은 scoring function $$\psi:\mathcal{E}\times\mathcal{R}\times\mathcal{E}\mapsto\Bbb R $$을 평가하는 포인트 단위 학습으로 형식화 될 수 있다. 
주어진 입력 트리플 $$x=\left(s,r,o\right)$$과 그 점수$$\psi\left(x\right)\in\Bbb R $$는 $$x$$가 참일 가능성에 비례한다.    

### Neural Link Predictors 
&nbsp;&nbsp;Neural link prediction 모델은 *encoding component*와 *scoring component*를 포함하는 multi-layer neural network로 볼 수 있다. 
입력 트리플 $$\left(s,r,o\right)$$이 주어지면 encoding component는 주어와 목적어 개체를 분산 embedding 표현 $$e_s,e_o\in\Bbb R^k$$으로 mapping 한다. 
scoring component에서는 두 개체 embedding이 $$\psi_r$$ 함수에 의해 평가된다. 
트리플의 점수는 $$\psi\left(s,r,o\right) = \psi_r\left(e_s,e_o\right)\in\Bbb R^k$$과 같이 정의된다.    

## Convolutional 2D Knowledge Graphs Embeddings
&nbsp;&nbsp;이 연구에서는 convolution과 fully-connected layer로 구성된 neural link prediction 모델을 제안한다. 
이 모델은 입력 개체와 관계 사이의 상호작용을 알아낸다. 
2D embedding을 convolution 연산하여 score를 정의하는 것이 이 모델의 특징이다.
이 모델의 구조는 아래 그림과 같다.    
|![그림1](/assets/images/convE_figure1.png "그림1"){: .align-center}|
|:---:|
|그림1: 개체와 관계 embedding이 reshape, concatenate 된다.(step1,2); 
그 행렬이 convolution layer에 입력으로 들어간다.(step3); 
결과로 나온 feature map tensor는 k-차원 공간으로 vectorised, projected 된다.(step4); 
목적어 embedding의 후보들과 match된다.(step5)|
  
scoring function은 다음과 같이 정의된다.  
$$\psi_r\left(e_s,e_o\right)\ =f(vec(f([\overline{e_s};\overline{r_r}]\ast \omega))W)e_o$$  
$$r_r\in\Bbb R^k$$는 $$r$$의 관계 parameter이다. 
$$\overline{e_s},\overline{r_r}$$는 $$e_s,r_r$$의 2D reshape를 표현한 것이다. 
$$e_s,r_r\in\Bbb R^k$$이면 $$\overline{e_s},\overline{r_r}\in\Bbb R^{k_w\times k_h}$$, $$k=k_w k_h$$이다.  
&nbsp;&nbsp;feed-forward pass에서 모델은 개체와 관계 두 embedding 행렬에 대한 row-vector look-up 연산을 수행한다. 
모델은 $$\overline{e_s},\overline{r_r}$$를 concatenate하여 $$\omega$$를 필터로 갖는 2D convolution layer에 입력한다. 
이 결과로 feature map tensor $$\mathcal{T}\in\Bbb R^{c\times m\times n}$$, $$c,m,n$$은 각각 2D feature map의 수와 그 차원이다. 
tensor $$\mathcal{T}$$는 vector$$\ vec(\mathcal{T})\in\Bbb R^{cmn}$$로 reshape 된다. 
행렬 $$W\in \Bbb R^{cmn\times k}$$로 선형변환되어 $$k$$차원의 공간으로 투영된다.
그 값은 목적어 embedding $$e_o$$와 내적으로 match 된다.  
&nbsp;&nbsp;모델의 parameter를 훈련시키기 위해서 score fucntion에 logistic sigmoid function을 적용하고, 아래의 binary cross entropy loss를 최소화하도록 하였다.  
$$\mathcal{L}(p,t) = -\frac{1}{N}\sum_i(t_i\cdot \log{(p_i)}+(1-t_i)\cdot\log{(1-p_i)})$$  
$$t$$는 1-1 scoring에서 $$\mathcal{R}^{1x1}$$차원 또는 1-N scoring에서 $$\mathcal{R}^{1xN}$$ 차원을 가지는 label vector이다. 
vector $$t$$의 요소는 존재하는 관계에 대해서는 1이고 다른 경우에는 0이다.  
&nbsp;&nbsp;모델의 빠른 훈련을 위해 비선형 $$f$$에 대해 rectified linear units을 사용했다. 
각 layer 마다 batch normalisation을 사용했다. 
dropout을 사용해 모델을 정규화했다. 
특히 embedding, feature map, convolution 연산, fully connected layer 이후의 hidden units에 dropout을 적용했다. 
Adam optimiser를 사용했고, label smoothing을 사용했다.    

### Fast Evaluation for Link Prediction Tasks
&nbsp;&nbsp;이 구조에서 convolution 연산은 전체 연산 시간의 75-90%을 차지한다. 
따라서 전체 연산 시간을 줄이려면 convolution 연산의 수를 줄이는 것이 중요하다. 
link prediction 모델에서는 평가 속도를 높이기 위해 주로 batch size를 증가시킨다. 
그러나 convolution 모델에서는 batch size를 늘리는데 필요한 메모리량이 GPU의 메모리 용량을 빠르게 넘어가기 때문에 적합하지 않다.  
&nbsp;&nbsp;ConvE는 다른 link prediction 모델과 같이 트리플$$(s,r,o)$$로 1-1 scoring을 하는 대신, 하나의 $$(s,r)$$쌍으로 모든 개체$$o\in \mathal{E}$$에 대해 1-N scoring을 시행한다.
convolution 모델의 평가에서 1-1 scoring과 1-N scoring은 약 300배의 시간 차이가 난다.    

## 실험
### Knowledge Graph Datasets
&nbsp;&nbsp;ConvE의 평가를 위해 다음과 같은 dataset을 사용했다.
+WN18 : 18개의 관계와 40,943개의 개체로 이뤄진 WordNet의 부분 집합, 151,442개의 트리플 대부분은 상위어와 하위어로 구성되어 있어 계층적인 구조이다.
+FB15k : 1,345개의 관계와 14,951개의 개체로 이뤄진 Freebase의 부분 집합, 이 그래프는 영화, 배우, 상, 스포츠, 스포츠 팀에 대한 설명으로 이뤄짐.
+YAGO3-10 : 37개의 관계와 123,182개의 개체로 이뤄진 YAGO3의 부분 집합, 각각 최소 10개의 관계를 갖는 개체로 구성됨.
+Countries
&nbsp;&nbsp;WN18과 FB15k는 역관계(inverse relation)으로 인해 test에 문제가 있다. 
training set의 트리플을 반전시킴으로써 간단히 많은 양의 test triple을 얻을 수 있다는 점이다. 
예를 들어 test set에 존재하는 (s, 하위어, o)인 트리플을 반전시킨 (o, 상위어, s)인 트리플이 training set에 있는 경우가 많다.  
&nbsp;&nbsp;논문에서는 간단한 룰 기반의 모델을 만들어 WN18과 FB15k에서 state-of-the-art 결과를 보임으로 이 문제를 제기하였다. 
역관계로 인한 test의 오류가 없는 것을 증명하기 위해 FB15k에서 역관계를 제거한 FB15k-237과 WN18에서 역관계를 제거한 WN18RR에 룰 기반 모델을 적용했다. 
이 논문에서는 앞으로의 연구에서 FB15k와 WN18 대신 YAGO3-10, FB15k-237과 WN18RR을 사용하는 것을 권장한다.    

### Experimental Setup
&nbsp;&nbsp;Validation set의 MRR에 대한 grid search를 통해 ConvE의 hyperparameter를 선택하였다. 
Grid search에 사용된 Hyperparameter의 범위는 다음과 같다.  
|Hyperparameter|Range|
|:---:|:---:|
|embedding dropout|0.0, 0.1, 0.2|
|feature map dropout|0.0, 0.1, 0.2, 0.3|
|projection layer dropout|0.0, 0.1, 0.3, 0.5|
|embedding size|100, 200|
|batch size|64, 128, 256|
|learning rate|0.001, 0.003|
|label smoothing|0.0, 0.1, 0.2, 0.3|
&nbsp;&nbsp;Hyperparameter에 대한 grid search외에도 2D Convolution layer를 fully connected layer 또는 1D convolution으로 대체하는 실험도 진행하였다. 
그러나 이러한 실험 결과는 좋지 않았다. 
또한 필터의 크기를 달리한 실험도 진행하였다. 
결과적으로 작은 크기(3x3)의 필터를 사용하는 경우에 좋은 결과를 얻었다.    

### Inverse Model
&nbsp;&nbsp;