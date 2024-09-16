# Amnesic Probing With a Single Projection
This repository contains the code to reproduce the results for the *Improving Causal Interventions in Amnesic Probing with Mean Projection* paper.
With Amnesic Probing approach we try to answer the following question:

*Is the target property of our interest important for a model to solve a given task?*

To verify this claim, we can try to remove the target property as precisely as possible and see if the performance of a model varies. If the performance drops significantly after the removal, we can suspect that the information was important, taking into account control tasks. However, if too much information is removed, the result is not reliable. This repository tests two methods for removing target information: INLP and Mean Projection (MP).

## References to other repositories
This repository contains additional code as well as original or modified code from the following repositories:
 
- https://github.com/yanaiela/amnesic_probing
- https://github.com/shauli-ravfogel/nullspace_projection
- https://github.com/tue-alga/debias-mean-projection

## Setup preparation
- This code has been tested using Python 3.7 and the conda environment on Linux. However, some modifications can allow the code to run in Python3.10, which is not currently included in this version of the project.
- You can use the *requirements.txt* file to set up the Python environment.
- To run the code directly without applying any modifications you need to add data to the *datasets* folder that should have the following structure:

```
.
├── ud_data_masked
│   ├── dev
│   │   ├── dep.pickle
│   │   ├── last_vec.npy
│   │   ├── pos.pickle
│   │   ├── tag.pickle
│   │   └── tokens.pickle
│   ├── test
│   │   ├── dep.pickle
│   │   ├── last_vec.npy
│   │   ├── pos.pickle
│   │   ├── tag.pickle
│   │   └── tokens.pickle
│   └── train
│       ├── dep.pickle
│       ├── last_vec.npy
│       ├── pos.pickle
│       ├── tag.pickle
│       └── tokens.pickle
└── ud_data_normal
    ├── dev
    │   ├── dep.pickle
    │   ├── last_vec.npy
    │   ├── pos.pickle
    │   ├── tag.pickle
    │   └── tokens.pickle
    ├── test
    │   ├── dep.pickle
    │   ├── last_vec.npy
    │   ├── pos.pickle
    │   ├── tag.pickle
    │   └── tokens.pickle
    └── train
        ├── dep.pickle
        ├── last_vec.npy
        ├── pos.pickle
        ├── tag.pickle
        └── tokens.pickle
```

- You also need to create in a folder *results* another folder with a proper name to store the output results.

## Access to the data

- The original splits come from the Amnesic Probing paper https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00359/98091/Amnesic-Probing-Behavioral-Explanation-with and can be accessed here: https://nlp.biu.ac.il/~lazary/amnesic_probing/

## Running the code
The repository contains scripts to get all the results. Once you have set up the environment and placed all the necessary data in the correct folder, you can simply run the scripts one at a time in the order shown below:

a) **Removing the target attribute**

The *remove_attribute.py* file contains the code that allows to remove target information
using INLP and MP for comparison. To obtain all combinations, inside the file the
pipeline is run 6 times. This allows us to get the MP and INLP results for all three linguistic
capabilities (pos, tag, and dep) and also for all two setups (ud data masked and ud data normal). Here
we get the following information:
- Accuracy before any change
- Accuracy after removing target property with INLP and MP
- Number of directions removed for INLP and MP
- Rank of matrix before any change
- Rank of matrix after INLP and MP was applied

b) **Running the evaluation and control**

After the target linguistic capability is removed we want to also obtain additional
evaluation of the results. In *evaluate_and_control.py* file you can find the script that
can provide us with the following information:
- Accuracy for the random control for INLP and MP
- Accuracy for the dropout control for INLP and MP
- Average cosine similarity of the original data with data after applying INLP and MP
- The difference in tokens distribution after applying INLP and MP
- Results of next token prediction task before any change
- Results of next token prediction task after applying INLP and MP

c) **Running baseline selectivity**

Selectivity check is demanding in terms of computation. To run the basic selectivity after the
information removal we need to run the script in *get_vanilla_selectivity.py* file.

d) **Running selectivity after the modifications**

To get the selectivity after injecting back the gold information to the embedding we run a script in the *get_selectivity_control.py* file.



