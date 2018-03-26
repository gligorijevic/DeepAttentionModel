# Deep Attention Model

This repository contains implementation of the Deep Attention Model proposed in the following [paper](https://astro.temple.edu/~tuf28053/papers/gligorijevicSDM18.pdf): 

Gligorijevic, Dj., Stojanovic, J., Satz, W., Stojkovic, I., Schreyer, K., Del Portal, D., Obradovic, Z. (2018) " Deep Attention Model for Triage of Emergency Department Patients," Proceedings of the 2018 SIAM International Conference on Data Mining (SDM 2018), San Diego, CA, May 2018.

If you publish articles using this code in any form, please cite the aforementioned paper.

## Run

### Prerequisites
To run the model, you will need
```
Python3
Tensorflow
scikit-learn
```

### Running the tests
Run training using: 
```
sh RUN_trainDAM.sh
```
Run testing using: 
```
sh RUN_testDAM.sh
```

## Sample data
Sample dataset provided with the implementation is [amazon fine food reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews) and it is processed from: 

J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.

*Sharing ED triage notes data is unfortunately currently not possible.

## Implementation Issues

For any problems detected with the implementation, please file an issue on the project.
