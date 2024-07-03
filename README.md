# ML summer projects
Repository for summer work at Imperial under supervision of Daneil Platt and Daattavya Argarwal.

# Prep-work/ Project 1
Use group invariant ML on a dataset from 𝐺2-geometry, try to get better accuracies than random guessing.
The files: prepworkSasakian.py and deep_sets.py involve implementing NN to learn Sasakian Hodge numbers and CN invariant [(1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub). Deep sets [(2)](https://arxiv.org/abs/1703.06114) are then introduced to ensure permutation invariance of the input vector; accuracies did not increase.

The topological invariants are in file: Topological_Data.txt giving the weights for the Sasakian Hodge numbers and CN invariant.

# Project 2
Does there exist a metric on the 3-dimensional torus 𝑇 3 such that every harmonic 1-form has a
zero?

Initial guess: yes. Goal of project is to produce numerical evidence for Yes/No. Define some metric on 𝑇3, compute the harmonic 1-forms and check if they have zeros. 

Initial work: ode_solver.py is not working correctly for now, was an attempt to reproduce students PINN [] to solve the ODE f''(x)=sinx but theirs was for a more complicated equation. Developed simpler model for this case shown in ode_simplesolver.py.
Install packages in requirements.txt. Tested with Python 3.9.6.

# References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114)
