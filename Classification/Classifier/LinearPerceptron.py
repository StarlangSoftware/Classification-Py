from Classification.Classifier.Classifier import Classifier
from Classification.InstanceList.InstanceList import InstanceList
from Classification.InstanceList.Partition import Partition
from Classification.Model.LinearPerceptronModel import LinearPerceptronModel
from Classification.Parameter.LinearPerceptronParameter import LinearPerceptronParameter


class LinearPerceptron(Classifier):

    def train(self,
              trainSet: InstanceList,
              parameters: LinearPerceptronParameter):
        """
        Training algorithm for the linear perceptron algorithm. 20 percent of the data is separated as cross-validation
        data used for selecting the best weights. 80 percent of the data is used for training the linear perceptron with
        gradient descent.

        PARAMETERS
        ----------
        trainSet : InstanceList
            Training data given to the algorithm
        parameters : LinearPerceptronParameter
            Parameters of the linear perceptron.
        """
        partition = Partition(instanceList=trainSet,
                              ratio=parameters.getCrossValidationRatio(),
                              seed=parameters.getSeed(),
                              stratified=True)
        self.model = LinearPerceptronModel(trainSet=partition.get(1),
                                           validationSet=partition.get(0),
                                           parameters=parameters)

    def loadModel(self, fileName: str):
        """
        Loads the linear perceptron model from an input file.
        :param fileName: File name of the linear perceptron model.
        """
        self.model = LinearPerceptronModel(fileName)
