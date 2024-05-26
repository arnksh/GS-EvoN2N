# GS-EvoN2N
Gauided Sampling-based Evolutionary Net2Net for architecture optimization

link to paper: https://www.sciencedirect.com/science/article/pii/S0952197623016822
Cite:

Arun K. Sharma, Nishchal K. Verma, "Guided sampling-based evolutionary deep neural network for intelligent fault diagnosis," Engineering Applications of Artificial Intelligence, Volume 128, 2024, 107498, ISSN 0952-1976,
https://doi.org/10.1016/j.engappai.2023.107498.
(https://www.sciencedirect.com/science/article/pii/S0952197623016822)
Abstract: The selection of model architecture and hyperparameters has a significant impact on the diagnostic performance of most deep learning models. Because training and evaluating the various architectures of deep learning models is a time-consuming procedure, manual selection of model architecture becomes infeasible. Therefore, we have proposed a novel framework for evolutionary deep neural networks that uses a policy gradient to guide the evolution of the DNN architecture towards maximum diagnostic accuracy. We have formulated a policy gradient-based controller that generates an action to sample the new model architecture at every generation so that optimality is obtained quickly. The fitness of the best model obtained is used as a reward to update the policy parameters. Also, the best model obtained is transferred to the next generation for quick model evaluation in the NSGA-II evolutionary framework. Thus, the algorithm gets the benefits of fast non-dominated sorting as well as quick model evaluation. The effectiveness of the proposed framework has been validated on three datasets: the Air Compressor dataset, the Case Western Reserve University dataset, and the Paderborn University dataset.

Keywords: Neural architecture search; Intelligent fault diagnosis; Deep neural network; Non-dominated sorting algorithm; Policy gradient


Code for Benchmark methods:
DNN: https://github.com/ArabelaTso/DeepFD DANN: https://github.com/NaJaeMin92/pytorch-DANN DTL: https://github.com/Xiaohan-Chen/transfer-learning-fault-diagnosis-pytorch DAFD: https://github.com/zggg1p/A-Domain-Adaption-Transfer-Learning-Bearing-Fault-Diagnosis-Model-Based-on-Wide-Convolution-Deep-Neu EvoDCNN: https://github.com/yn-sun/evocnn psoCNN: https://github.com/feferna/psoCNN
