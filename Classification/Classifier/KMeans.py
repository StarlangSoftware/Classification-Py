from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.KMeansModel import KMeansModel
from Classification.Parameter.KMeansParameter import KMeansParameter


class KMeans(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: KMeansParameter):
        prior_distribution = trainSet.classDistribution()
        class_means = InstanceList()
        class_lists = Partition(trainSet)
        for i in range(class_lists.size()):
            class_means.add(class_lists.get(i).average())
        self.model = KMeansModel(priorDistribution=prior_distribution,
                                 classMeans=class_means,
                                 distanceMetric=parameters.getDistanceMetric())
