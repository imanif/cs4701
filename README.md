# cs4701
## Conversion of LAST library from JAX to PyTorch

Today, end-to-end audio speech recognition models have become popular as they show significant word error rate (WER) reduction with more flexible models and frameworks. While end-to-end audio speech recognition models utilize a simplified pipeline, their rise has come at the expense of black-box formulation and lack of explainability and interpretability. Hence probabilistic modeling of modern neural ASR aims to address these problems. 
Earlier this year, Wu et al. introduced a LAttice-based Speech Transducer (LAST) library in JAX. LAST implements a differentiable weighted finite state automata (WFSA) framework. The library presents probabilistic modeling of audio-speech recognition (ASR) in three components –– context dependency, alignment lattice, and weight function –– based on global normalization for streaming speech recognition (GNAT). 
This project introduces PyLAST, a LAttice-based Speech Transducer library in PyTorch. The overall aim is to expand the accessibility of LAST by offering the library in PyTorch, a machine learning framework with similar functionality to JAX but is more familiar with machine learning communities. Specifically, PyTorch has modules for customizable gradient functions and neural networks that can be used for the weight function and alignment lattice.
Evaluation
The goal of this project is to develop a software package that implements weighted finite state transducers for richer speech representation within ASR models. Referencing the JAX library, I will develop a version using the PyTorch framework and NumPy API. The library designed, like the original, will be guided by GNAT. The Github for JAX includes test cases that, once converted into PyTorch-compatible versions, will be used for evaluation.

### Timeline
**Due Oct 13**
- [x] ```semirings.py```
- [x] ```semirings_test.py```
- [x] ```alignments.py```
- [x] ```alignments_test.py```
- [x] ```contexts.py```
- [x] ```contexts_test.py```

**Due Oct 20**
- [x] ```weight_fn.py```
- [x] ```weight_fn_test.py```

**Due Oct 27**
- [ ] ```lattices.py```
- [ ] ```lattices_test.py```

**Due Nov 3**
- [ ] Build a simple speech recognition model with [spoken digit dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)

**Due Nov 10**
- [ ] Train 1 locally normalized and 1 globally normalized model

**Due Nov 17** 
- [ ] Evaluate and compare global and local normalization models
