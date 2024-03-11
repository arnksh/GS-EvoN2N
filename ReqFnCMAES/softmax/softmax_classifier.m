function  [saeOptTheta] = softmax_classifier(hiddenSize, numClasses, ...
    				         trainFeatures, trainLabels)



softmaxModel = struct;
options2.maxIter = 100;
lambda = 1e-4;
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
			    trainFeatures, trainLabels, options2);
saeOptTheta = softmaxModel.optTheta(:);

end
