# Classification

## Classification Algorithms

1. Dummy: All test instances are assigned to the class with the maximum prior.
2. C45: The archetypal decision tree method.
3. Knn: K-Nearest Neighbor classification algorithm that uses the Euclidean distance.
4. Rocchio: Nearest-mean classification algorithm that uses the Euclidean distance.
5. Linear Perceptron: Linear perceptron with softmax outputs trained by gradient-descent to minimize cross-entropy.
6. Multi Layer Perceptron: Well-known multilayer perceptron classification algorithm.
7. Naive Bayes: Classic Naive Bayes classifier where each feature is assumed to be Gaussian distributed and each feature is independent from other features.
8. RandomForest: Random Forest method improves bagging idea with randomizing features at each decision node and called these random decision trees as weak learners. In the prediction time, these weak learners are combined using committee-based procedures.

For Developers
============
You can also see [Java](https://github.com/olcaytaner/Classification), [C++](https://github.com/olcaytaner/Classification-CPP), or [C#](https://github.com/olcaytaner/Classification-CS) repository.

## Requirements

* [Python 3.7 or higher](#python)
* [Git](#git)

### Python 

To check if you have a compatible version of Python installed, use the following command:

    python -V
    
You can find the latest version of Python [here](https://www.python.org/downloads/).

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called Classification will be created. Or you can use below link for exploring the code:

	git clone https://github.com/olcaytaner/Classification-Py.git

## Open project with Pycharm IDE

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `Classification-Py` file
* Select open as project option
* Couple of seconds, dependencies will be downloaded. 


## Compile

**From IDE**

After being done with the downloading and Maven indexing, select **Build Project** option from **Build** menu. After compilation process, user can run Classification-Py.

Detailed Description
============
+ [Classification Algorithms](#classification-algorithms)
+ [Sampling Strategies](#sampling-strategies)
+ [Feature Selection](#feature-selection)
+ [Statistical Tests](#statistical-tests)

## Classification Algorithms

Algoritmaları eğitmek için

	train(self, trainSet: InstanceList, parameters: Parameter)

Eğitilen modeli bir veri örneği üstünde sınamak için

	predict(self, instance: Instance) -> str

Karar ağacı algoritması C45 sınıfında

Bagging algoritması Bagging sınıfında

Derin öğrenme algoritması DeepNetwork sınıfında

KMeans algoritması KMeans sınıfında

Doğrusal ve doğrusal olmayan çok katmanlı perceptron LinearPerceptron ve 
MultiLayerPerceptron sınıflarında

Naive Bayes algoritması NaiveBayes sınıfında

K en yakın komşu algoritması Knn sınıfında

Doğrusal kesme analizi algoritması Lda sınıfında

İkinci derece kesme analizi algoritması Qda sınıfında

Destek vektör makineleri algoritması Svm sınıfında

RandomForest ağaç tabanlı ensemble algoritması RandomForest sınıfında

Basit dummy ve rasgele sınıflandırıcı gibi temel iki sınıflandırıcı Dummy ve 
RandomClassifier sınıflarında

## Sampling Strategies

K katlı çapraz geçerleme deneyi yapmak için KFoldRun, KFoldRunSeparateTest, 
StratifiedKFoldRun, StratifiedKFoldRunSeparateTest

M tane K katlı çapraz geçerleme deneyi yapmak için MxKFoldRun, MxKFoldRunSeparateTest,
StratifiedMxKFoldRun, StratifiedMxKFoldRunSeparateTest

Bootstrap tipi deney yapmak için BootstrapRun

## Feature Selection

Pca tabanlı boyut azaltma için Pca sınıfı

Discrete değişkenleri Continuous değişkenlere çevirmek için DiscreteToContinuous sınıfı

Discrete değişkenleri binary değişkenlere değiştirmek için LaryToBinary sınıfı

## Statistical Tests

İstatistiksel testler için Combined5x2F, Combined5x2t, Paired5x2t, Pairedt, Sign sınıfları
