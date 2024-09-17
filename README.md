# Group Invariant ML summer project
Repository for summer work Project 1 at Imperial in collaboration with Hugo Robijns, under supervision of Daniel Platt and Daattavya Argarwal.

## Packages and Requirements
Install packages in requirements.txt. Tested with Python 3.9.6.

## Prep-work/ Project 1
Use group invariant ML on a dataset from 𝐺2-geometry, try to get better accuracies than random guessing by experimenting with different architecture and introducing group invariant ML. Original results published in [this paper (1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub).
The files: Sasakian_vanilla.py and Sasakian_deep_sets.py involve implementing a simple NN to learn Sasakian Hodge numbers and CNI_vanilla.py and CNI_deepsets.py involve a classification NN to learn the CN invariant [(1)], both from their weights.

Deep sets from Lemma 3 (Section 2.2, page 2) [(2)](https://arxiv.org/abs/1703.06114) are introduced to ensure permutation invariance of the input vector; accuracies did not increase. Group invariance also enforced through 1a): train a NN on our dataset then average over the 120 permutations, 1b): spliting the input vector into its 120 permutations then using these permutations to train a NN in parallel and finally summing together.

Implementing group invariant architectures did not improve the accuracy (deep sets led to a decrease in accuracy). Results are summarised [in this report (4)].


## References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114)
(3) [Platt et al., 2022](https://openreview.net/pdf?id=RLkbkAgNA58) <br/>
(4) [Report](https://github.com/mdh959/Imperial_projects2024/blob/main/report.tex)
