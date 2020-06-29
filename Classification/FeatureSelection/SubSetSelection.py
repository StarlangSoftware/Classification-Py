from abc import abstractmethod
from copy import deepcopy

from Classification.Experiment.Experiment import Experiment
from Classification.Experiment.MultipleRun import MultipleRun
from Classification.FeatureSelection.FeatureSubSet import FeatureSubSet


class SubSetSelection(object):

    initialSubSet: FeatureSubSet

    def __init__(self, initialSubSet: FeatureSubSet):
        """
        A constructor that sets the initial subset with given input.

        PARAMETERS
        ----------
        initialSubSet : FeatureSubSet
            FeatureSubSet input.
        """
        self.initialSubSet = initialSubSet

    @abstractmethod
    def operator(self, current: FeatureSubSet, numberOfFeatures: int) -> list:
        pass

    def forward(self, currentSubSetList: list, current: FeatureSubSet, numberOfFeatures: int):
        """
        The forward method starts with having no feature in the model. In each iteration, it keeps adding the features
        that are not currently listed.

        PARAMETERS
        ----------
        currentSubSetList : list
            List to add the FeatureSubsets.
        current : FeatureSubSet
            FeatureSubset that will be added to currentSubSetList.
        numberOfFeatures : int
            The number of features to add the subset.
        """
        for i in range(numberOfFeatures):
            if not current.contains(i):
                candidate = deepcopy(current)
                candidate.add(i)
                currentSubSetList.append(candidate)

    def backward(self, currentSubSetList: list, current: FeatureSubSet):
        """
        The backward method starts with all the features and removes the least significant feature at each iteration.

        PARAMETERS
        ----------
        currentSubSetList : list
            List to add the FeatureSubsets.
        current : FeatureSubSet
            FeatureSubset that will be added to currentSubSetList
        """
        for i in range(current.size()):
            candidate = deepcopy(current)
            candidate.remove(i)
            currentSubSetList.append(candidate)

    def execute(self, multipleRun: MultipleRun, experiment: Experiment) -> FeatureSubSet:
        """
        The execute method takes an Experiment and a MultipleRun as inputs. By selecting a candidateList from given
        Experiment it tries to find a FeatureSubSet that gives best performance.

        PARAMETERS
        ----------
        multipleRun : MultipleRun
            MultipleRun type input.
        experiment : Experiment
            Experiment type input.

        RETURNS
        -------
        FeatureSubSet
            FeatureSubSet that gives best performance.
        """
        processed = set()
        best = self.initialSubSet
        processed.add(best)
        betterFound = True
        bestPerformance = None
        if best.size() > 0:
            bestPerformance = multipleRun.execute(experiment.featureSelectedExperiment(best))
        while betterFound:
            betterFound = False
            candidateList = self.operator(best, experiment.getDataSet().getDataDefinition().attributeCount())
            for candidateSubSet in candidateList:
                if candidateSubSet not in processed:
                    if candidateSubSet.size() > 0:
                        currentPerformance = multipleRun.execute(experiment.featureSelectedExperiment(candidateSubSet))
                        if bestPerformance is None or currentPerformance.isBetter(bestPerformance):
                            best = candidateSubSet
                            bestPerformance = currentPerformance
                            betterFound = True
                    processed.add(candidateSubSet)
        return best
