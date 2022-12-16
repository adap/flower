---
name: Baseline request
about: Suggest a new baseline
title: Add [BASELINE_NAME] baseline
labels: 'new baseline'
assignees: ''

---

### What paper would you like to be implemented as a baseline?

- Author(s): <!-- Replace by the author(s) of the paper -->
- Year: <!-- Replace by the year the paper was published in -->
- Title: <!-- Replace by the title of the paper -->
- Abstract: <!-- Replace by the link (ideally an arxiv.org/abs/* link) to the abstract of the paper -->

### Why would you like this baseline to be implemented?
<!-- Quickly give reasons why, if any, this paper should be implemented before others. 
Otherwise, you can just remove this section.
-->





<!-- Leave everything below untouched -->
<details>
    <summary><h3> For first time contributers</h3></summary>

- [ ]  Read the [`first contribution` doc](TODO)
- [ ]  Complete the Flower tutorial
- [ ]  Read the Flower Baselines docs to get an overview:
    - [ ]  [https://flower.dev/docs/using-baselines.html](https://flower.dev/docs/using-baselines.html)
    - [ ]  [https://flower.dev/docs/contributing-baselines.html](https://flower.dev/docs/contributing-baselines.html)

</details>

### Implementation 

#### Prep - understand the scope

It’s recommended to do the following items in that order:

- [ ]  Read the paper linked above
- [ ]  Create the directory structure in Flower Baselines (just the `__init__.py`files and a README)
- [ ]  Before starting to write code, write down all of the specs of this experiment in a README (dataset, partitioning, model, number of clients, all hyperparameters, …)
- [ ]  Open a draft PR

#### Implement - make it work

Everything up to this point should be pretty mechanical, the goal is to get a simple version of this working as quickly as possible, which means a model that’s starting to converge (doesn’t have to be good).

- [ ]  Implement some form of dataset loading and partitioning in a separate `dataset.py` (doesn’t have to match the paper exactly)
- [ ]  Implement the model in PyTorch
- [ ]  Write a test that shows that the model has the number of parameters mentioned in the paper
- [ ]  Implement the federated learning setup outlined in the paper, maybe starting with fewer clients
- [ ]  Plot accuracy and loss
- [ ]  Run it and check if the model starts to converge

#### Align - make it correct

- [ ]  Implement the exact data partitioning outlined in the paper
- [ ]  Use the exact hyperparameters outlined in the paper

#### Tuning - make it converge

- [ ]  Make it converge to roughly the same accuracy that the paper states
- [ ]  Mark the PR as ready
