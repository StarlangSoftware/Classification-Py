from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='NlpToolkit-Classification',
    version='1.0.15',
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
    install_requires=['NlpToolkit-Math', 'NlpToolkit-DataStructure', 'NlpToolkit-Sampling', 'NlpToolkit-Util'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
