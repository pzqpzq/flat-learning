# Flat Learning
This project includes portions of code related to Flat Learning.

In the dynTest/old_codes directory, we release the main codes of our published paper (accepted as an oral paper at ICML-2023, url: https://icml.cc/virtual/2023/oral/25562 ）that introduces the DyN (Dynamics-inspired Neuromorphic) architecture. 

Any_LinearTransform_via_dyn explains how to implement the fitting process of any linear mapping (matrix) through a dynamical approach. 

LeNet_toyTrainer is an initial toy code that explains how to convert the LeNet network into a DyN form with fewer parameters, and it's tested on the MNIST dataset (please note, for compatibility with CUDA, we need to revert the DyN system back to the form of a weight matrix here). 

The general_fastFC section discusses how to carry out a real inference process (matrix multiplication) without the need for a weight matrix, meaning, we do not need to revert the DyN system back to a weight matrix, but directly use the DyN system itself as an input to complete the matrix multiplication operation (please note, this version currently only supports CPU, future releases will gradually introduce a GPU version based on CUDA).

Other codes related to large-scale models and dyn-training from scratch will be introduced in future releases.

In the KasF (Knowledge as Function) directory, we will release a knowledge inference system based on flattened dynamics. The relevant system (Albert+KasF) achieved first place in both single and multi-model categories in the CommonsenseQA leaderboard (https://www.tau-nlp.sites.tau.ac.il/csqa-leaderboard).
