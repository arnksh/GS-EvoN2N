function netOpt = softmax_finetuning(aeOptTheta,hiddenSizes,inputSize, ...
                                                                numClasses, trainFeatures,trainData,trainLabels)	 
       
           


    %%======================================================================
	%Step 1.  %% Softmax_classifier
    %%======================================================================
        
        [softmaxTheta] = softmax_classifier(hiddenSizes(end), numClasses, ...
                                           trainFeatures, trainLabels);



        
    %%======================================================================
	%step 2. %% finetuning
    %%======================================================================
        
    %% STEP 6: Finetune softmax model
    
    stack = cell(length(hiddenSizes),1);
    n = [inputSize hiddenSizes];
    for i = 1:length(hiddenSizes)
        stack{i}.w = reshape(aeOptTheta{i}(1:n(i+1)*n(i)), n(i+1), n(i));
        stack{i}.b = aeOptTheta{i}(n(i+1)*n(i)+1:n(i+1)*n(i)+n(i+1));
    end
    
    % Initialize the parameters for the deep model
    [stackparams, netconfig] = stack2params(stack);
    netconfig.numClasses = numClasses;
    stackedAETheta = [softmaxTheta; stackparams]; %Initial theta0 (@p) for the minFunc optimizer
   
    options.MaxIter = 400; %250
    options.Display = 'off';
    lambda2 = 1e-5;
    [stackedAEOptTheta, ~] = minFunc( @(p) stackedAECost(p, netconfig,lambda2, ...
        trainData, trainLabels), stackedAETheta, options);
    
    %======================== convertion to net model===================
    % Extract weights parameters
    softmaxOptTheta = reshape(stackedAEOptTheta(1:hiddenSizes(end)*numClasses), ...
                                                    numClasses, hiddenSizes(end));
    stack = params2stack(stackedAEOptTheta(hiddenSizes(end)*numClasses+1:end), netconfig);

    
    netOpt.W = {}; netOpt.b = {};
    for d = 1:numel(stack)
        netOpt.W{d} = stack{d}.w;
        netOpt.b{d} = stack{d}.b;
    end
    netOpt.W{numel(stack)+1} = softmaxOptTheta;
    netOpt.b{numel(stack)+1} = zeros(numClasses,1);
    netOpt.nh = netconfig.nh;
end   

