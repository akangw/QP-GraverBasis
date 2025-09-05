<<<<<<< HEAD
# GPU-Based Graver Basis Extraction for Nonlinear Integer Optimization

## Introduction

Nonlinear integer programs entail optimizing nonlinear objective functions where the decision variables are constrained to integer values.  In this work, we are particularly concerned with problems that are subject to linear constraints, which can be formulated as follows:

$$
\begin{aligned}
&\underset{x\in\mathbb Z^n}{\min} && f(x)\\
    &\text{s.t.} && Ax=b\\
    & && l\leq x\leq u,
\end{aligned}
$$

where $f:\mathbb R^n\rightarrow\mathbb R$ is a real-valued function, and $A\in\mathbb Z^{m  \times n}, b\in\mathbb Z^m, l, u\in\mathbb{Z}^n$ define linear constraints and variable bounds. A key solution approach is the augmentation procedure, which iteratively improves incumbent solutions using directions from a precomputed test set. For example, $Ker_\mathbb Z(A)$ serves as a test set. A more compact test set is the Graver basis, defined as all the $\sqsubseteq-$ elements in $Ker_\mathbb Z(A)\backslash {0}$.  However, computing this basis exactly is computationally prohibitive. We propose Multi-start Augmentation via Parallel Extraction (MAPE), a GPU-based heuristic to efficiently approximate the Graver Basis. MAPE extracts test directions by solving non-convex continuous problems via parallelizable first-order methods, then uses these directions in multi-start augmentations to refine solutions. 

## Software dependencies

```markdown
python=3.11
pytorch=2.1.0 
hsnf=0.3.16
```

## Running experiment

Use the following command to run MAPE on QPLIB instances:

```markdown
python mape.py
```

Use the following command to run baseline solvers on QPLIB instances:

```markdown
python baseline_solvers.py --solver 'gurobi'
python baseline_solvers.py --solver 'scip'
python baseline_solvers.py --solver 'cplex'
```

## Citing our work
If you would like to use this repository, please consider citing this work using the following Bibtex:
```text
@article{liu2024gpu,
  title={GPU-Based Graver Basis Extraction for Nonlinear Integer Optimization},
  author={Liu, Wenbo and Wang, Akang and Yang, Wenguo},
  journal={arXiv preprint arXiv:2412.13576},
  year={2024}
}
```

=======
# QP-GraverBasis
>>>>>>> 51e751d6248a38679382ea0a210018f7f4fa0be0
