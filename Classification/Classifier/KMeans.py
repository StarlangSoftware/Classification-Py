from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.KMeansModel import KMeansModel
from Classification.Parameter.KMeansParameter import KMeansParameter


class KMeans(Classifier):

    def train(self, trainSet: InstanceList, parameters: KMeansParameter):
        priorDistribution = trainSet.classDistribution()
        classMeans = InstanceList()
        classLists = Partition(trainSet)
        for i in range(classLists.size()):
            classMeans.add(classLists.get(i).average())
        self.model = KMeansModel(priorDistribution, classMeans, parameters.getDistanceMetric())
