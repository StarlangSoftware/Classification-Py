from Classification.Instance.CompositeInstance import CompositeInstance
from Classification.Instance.Instance import Instance
from Classification.InstanceList.InstanceList import InstanceList
from Classification.Model.DecisionTree.DecisionNode import DecisionNode
from Classification.Model.ValidatedModel import ValidatedModel


class DecisionTree(ValidatedModel):

    """
    Constructor that sets root node of the decision tree.

    PARAMETERS
    ----------
    root : DecisionNode
        DecisionNode type input.
    """
    def __init__(self, root: DecisionNode):
        self.root = root

    """
    The predict method  performs prediction on the root node of given instance, and if it is null, it returns the 
    possible class labels. Otherwise it returns the returned class labels.

    PARAMETERS
    ----------
    instance : Instance
        Instance make prediction.
        
    RETURNS
    -------
    str
        Possible class labels.
    """
    def predict(self, instance: Instance) -> str:
        predictedClass = self.root.predict(instance)
        if predictedClass is None and isinstance(instance, CompositeInstance):
            predictedClass = instance.getPossibleClassLabels()
        return predictedClass

    """
    The prune method takes an InstanceList and  performs pruning to the root node.

    PARAMETERS
    ----------
    pruneSet : InstanceList
        InstanceList to perform pruning.
    """
    def prune(self, pruneSet: InstanceList):
        self.root.prune(self, pruneSet)
