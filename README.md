# ISCMF: Integrated Similarity-Constrained Matrix Factorization for Drug-Drug Interaction Prediction

In this study, we use Integrated Similarity-Constrained Matrix Factorization (ISCMF) to predict DDIs. Eight similarities based on the drug substructure, targets, side effects, off-label side effects, pathways, transporters, enzymes, and indication data as well as Gaussian interaction profile for the drug pairs are calculated. Subsequently, a non-linear similarity fusion method is used to integrate multiple similarities and make them more informative. Finally, we use ISCMF which projects the drugs in the interaction space into a low-rank space constrained to obtain new insight about DDIs.

**Link of paper**: https://link.springer.com/article/10.1007%2Fs13721-019-0215-3

![ISCMF schema](https://github.com/nrohani/ISCMF/blob/master/abstract.jpg)


### Dependency:
- python version 3.5.3
- scikit-learn
### Codes and Data
Find data in DS1 folder.
Codes of functions of ISCMF are available in code.py.

Find SNF code in https://github.com/nrohani/NDD/tree/master/NDD
### Contact
Please do not hesitate to contact me if you have any question: 

Mail: n.rohani@mail.sbu.ac.ir

Please cite us if you find this study helpful.
