# ML summer projects
Repository for summer work at Imperial under supervision of Daneil Platt and Daattavya Argarwal.

# Prep-work/ Project 1
Use group invariant ML on a dataset from 𝐺2-geometry, try to get better accuracies than random guessing by experimenting with different architecture and introducing group invariant ML. Original results published in [this paper (1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub).
The files: prepworkSasakian.py and deep_sets.py involve implementing a simple NN to learn Sasakian Hodge numbers and CN invariant [(1)] from their weights.

Deep sets [(2)](https://arxiv.org/abs/1703.06114) are then introduced to ensure permutation invariance of the input vector; accuracies did not increase. Group invariance also enforced through 1a): train a NN on our dataset then average over the 120 permutations, 1b): spliting the input vector into its 120 permutations then using these permutations to train a NN in parallel and finally summing together.

# Project 2
Does there exist a metric on the 3-dimensional torus 𝑇 3 such that every harmonic 1-form has a
zero?

Initial guess: yes. Goal of project is to produce numerical evidence for Yes/No. Define some metric on 𝑇3, compute the harmonic 1-forms and check if they have zeros. 

Initial work: Developed simple model for solving an ODE f''=sin(x), case shown in ode_simplesolver.py, used ideas displayed in [this paper (4)](https://arxiv.org/abs/1711.10561).


Install packages in requirements.txt. Tested with Python 3.9.6.

# References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114)
(3) [Platt et al., 2022](https://openreview.net/pdf?id=RLkbkAgNA58) <br/>
(4) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561)
