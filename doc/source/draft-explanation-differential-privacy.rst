Differential Privacy Explainer: From Fundamentals to Using it in Flower Federated Learning Framework
====================
The information in datasets like healthcare, financial transactions, user preferences, and etc. is valuable and has the potential for scientific breakthroughs and provide important business insights. However, such data is also sensitive and there is a risk of compromising individual privacy.
Traditional methods like anonymization alone would not work because of attacks like Re-identification and Data Linkage.
That's where differential privacy comes in. It provides the possibility of analyzing data while ensuring the privacy of individual.


Differential Privacy
-------
Imagine two datasets that are identical except for a single record (for instance Alice's data).
Differential Privacy (DP) guarantees that any analysis (M), like calculating the average income, will produce nearly identical results for both datasets (O and O' would be similar).
This preserves group patterns while obscuring individual details, ensuring individual's information remains hidden in the crowd.

.. image:: ./_static/DP/dp-intro.png
  :width: 400
  :alt: DP Intro


One of the most commonly used mechanisms achieve DP is adding enough noise to the output to mask the contribution of each individual in the data while still preserving the overall accuracy of the analysis.

Formal Definition
~~~~~~~~~~
Differential Privacy (DP) provides statistical guarantees against the information an adversary can infer through the output of a randomized algorithm.
It provides an unconditional upper bound on the influence of a single individual on the output of the algorithm by adding noise [1].
A randomized mechanism
M provides (:math:`\epsilon`, :math:`\delta`)-differential privacy if for any two neighboring databases, D :sub:`1` and D :sub:`2`, that differ in only a single record,
and for all possible outputs S ⊆ Range(A):

:math:`P[M(D_{1} \in A)] \leq e^{\delta} P[M(D_{2} \in A)] + \delta`


The :math:`\epsilon` parameter, also known as the privacy budget, is a metric of privacy loss.
It also controls the privacy-utility trade-off; lower :math:`\epsilon` values indicate higher levels of privacy but are likely to reduce utility as well.
The :math:`\delta` parameter accounts for a small probability on which the upper bound :math:`\epsilon` does not hold.
The amount of noise needed to achieve differential privacy is proportional to the sensitivity of the output, which measures the maximum change in the output due to the inclusion or removal of a single record.


Differential Privacy in Machine Learning
-------
Differentially private machine learning algorithms are designed in a way to protect the privacy of individuals data points in the training data.
There are different ways to apply DP to machine learning algorithms.
One common approach is to add noise du


Differential Privacy in Federated Learning
-------
Federated learning is a data minimization approach that allows multiple parties to collaboratively train a model without sharing their raw data.
However, federated learning also introduces new privacy challenges. The model updates between parties and the central server can leak information about the local data.
There are attacks that exploit such leaks, such as membership and property inference attacks or model inversion attacks.
Differential privacy can play a role in federated learning to provide privacy for the clients' data.

There are different forms that differential privacy can be integrated into federated learning.
It is the matter of granularity that the definition of DP can be applied.
We mainly categorize it into following categories.

- Central Differential Privacy

- tes


Central Differential Privacy
~~~~~~~~~~


Local Differential Privacy
~~~~~~~~~~


Distributed Differential Privacy
~~~~~~~~~~

Differential Privacy in Flower
-------
Note: we are at the experimental phase of using differential privacy in Flower. Please


[1] Dwork et al. The Algorithmic Foundations of Differential Privacy.
