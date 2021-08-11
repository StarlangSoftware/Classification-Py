from setuptools import setup

setup(
    name='NlpToolkit-Classification',
    version='1.0.10',
    packages=['Classification', 'Classification.Model', 'Classification.Model.DecisionTree', 'Classification.Filter',
              'Classification.DataSet', 'Classification.Instance', 'Classification.Attribute',
              'Classification.Parameter', 'Classification.Classifier', 'Classification.Experiment',
              'Classification.Performance', 'Classification.InstanceList', 'Classification.DistanceMetric',
              'Classification.StatisticalTest', 'Classification.FeatureSelection'],
    url='https://github.com/StarlangSoftware/Classification-Py',
    license='',
    author='olcaytaner',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='Classification library',
    install_requires=['NlpToolkit-Math', 'NlpToolkit-DataStructure', 'NlpToolkit-Sampling']
)
