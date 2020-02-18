from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.Model import Model
from Classification.Performance.ClassificationPerformance import ClassificationPerformance


class ValidatedModel(Model):

    def testClassifier(self, data: InstanceList) -> ClassificationPerformance:
        """
        The testClassifier method takes an InstanceList as an input and returns an accuracy value as
        ClassificationPerformance.

        PARAMETERS
        ----------
        data : InstanceList
            InstanceList to test.

        RETURNS
        -------
        ClassificationPerformance
            Accuracy value as ClassificationPerformance.
        """
        total = data.size()
        count = 0
        for i in range(data.size()):
            if data.get(i).getClassLabel() == self.predict(data.get(i)):
                count = count + 1
        return ClassificationPerformance(count / total)
