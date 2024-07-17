# Flat Learning
This project includes portions of code related to Flat Learning.

In the dynTest/old_codes directory, we release the main codes of our published paper (accepted as an oral paper at ICML-2023, url: https://icml.cc/virtual/2023/oral/25562 ）that introduces the DyN (Dynamics-inspired Neuromorphic) architecture. 

Any_LinearTransform_via_dyn explains how to implement the fitting process of any linear mapping (matrix) through a dynamical approach. 

LeNet_toyTrainer is an initial toy code that explains how to convert the LeNet network into a DyN form with fewer parameters, and it's tested on the MNIST dataset (please note, for compatibility with CUDA, we need to revert the DyN system back to the form of a weight matrix here). 

The general_fastFC section discusses how to carry out a real inference process (matrix multiplication) without the need for a weight matrix, meaning, we do not need to revert the DyN system back to a weight matrix, but directly use the DyN system itself as an input to complete the matrix multiplication operation (please note, this version currently only supports CPU, future releases will gradually introduce a GPU version based on CUDA).

In the LasF (Language tokens as Functional) directory, we release a novel flat language model introduced by our published paper (accepted as a poster paper at ICML-2024, url: https://icml.cc/virtual/2024/poster/34594), i.e., Modeling Language Tokens as Functionals of Semantic Fields. 
The proposed language model is competitive with the Transformer module of the same scale in language modelling task.
The relevant system, i.e., Albert+KasF (Knowledge as Functional), achieved first place in both single and multi-model categories in the CommonsenseQA leaderboard (https://www.tau-nlp.sites.tau.ac.il/csqa-leaderboard).

In the RieM (Riemannian Metric for neural models) directory, we release a straightforward data-free algorithm that can compress a weight matrix of a pre-trained model into neuronal dynamics via the proposed Riemannian metric.
This Neuronal Riemannian Metric is introduced by our published paper (accepted as an oral paper at ICML-2024, url: https://icml.cc/virtual/2024/oral/35536), i.e., Data-free Neural Representation Compression with Riemannian Neural Dynamics.

Contact with: peizhengqi22@mails.ucas.ac.cn

