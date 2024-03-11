clear all; clc
addpath ('./ReqFnCMAES'); addpath ('./ReqFnCMAES/softmax')
addpath ./ReqFnCMAES/minFunc/;
%%
Np      = 100;       % Number of chromosomes in the population
maxgen  = 20;       % Maximum number of generations
pc      = 0.8;       % Probability of crossover
pm      = 0.2;       % Probability of mutation
nVar    = 2;         % Number of variables (dimensions or objectives)
n_R = [1, 10];   %range for maximum depth
h_R = [10, 400]; % range of number of nodes in a layer
% (max<=dimention of input dataset and min >= numClass)

%%% Name of the data and Data Folder
dataFolder = 'GFD_40';
tarData = {'h_b_30hz_30', 'h_b_30hz_50', 'h_b_30hz_70', 'h_b_30hz_90'};

%% Initialize the variables to save results
% results = struct('gen', cell(1,maxgen+1), 'best_net', cell(1,maxgen+1), 'MSE', cell(1,maxgen+1));
acc_log = zeros(maxgen+1,1);
Ffit = cell(maxgen+1,1);
fileID = fopen('./logs_GFD/NSGA_net2net_400.txt','w');
%% Main loop

fprintf(fileID, '\n\n*****Target data: 400 samples/Class\n');
for ca=1:1%length(tarData)
    
    %     Y = train_test_val(dataFolder, tarData{ca});
    load(['./' dataFolder '/' tarData{ca}], 'Y');
    
    
    %% ==========Initialization====================
    load('./logs_GFD/TeacherNet', 'net')
    gen = 0;
    m =[mean(n_R), mean(h_R)]; sigma = [range(n_R)/2, range(h_R)/2];
    P = genPop(m, sigma, Np, n_R, h_R);   % Np number of different solution
    P{1} = [net.nh]; % Replace 1st chromosome with the teacher architecture
    [Pfit, P, best_net]  = n2n_feval(gen, net, P, Y); % Accuracy of selected network architecture
    fprintf('Gen: #%d, \t best model: Acc=%f\n\n', gen, best_net.ACC);
    fprintf('Values of directional trms, %f, %f\n', m, sigma)
    acc_log(1) = best_net.ACC;
    fprintf(fileID, '\t For tarData %s, best architecture at Gen %d: %s\n',...
         tarData{ca}, best_net.nh, gen);
    
    Ffit{gen+1} = Pfit;
    
    Prank = FastNonDominatedSorting_Vectorized(Pfit);
    [P,~] = selectParentByRank(P, Prank);
    [m, sigma] = update_m_sigma(gen, P, Ffit, m, sigma);
    fprintf('Values of directional trms, %f, %f\n\n', m, sigma)
    Q = applyCrossoverAndMutation(P, pc, pm, n_R, h_R, m, sigma);
    
    
    % ========================================================
    % NSGA-II loop (evolve through generations)
    termination = 0; gen = 1;
    while termination == 0 && gen <= maxgen
        % (i) Merge the parent and the children
        R = [P; Q];
        
        % (ii) Compute the new fitness and Pareto Fronts
        [Rfit, R, best_net]  = n2n_feval(gen, best_net, R, Y);
        Rrank = FastNonDominatedSorting_Vectorized(Rfit);
        fprintf('Gen: #%d:  FF size: %d,  best model: ACC=%f\n', gen, sum(Rrank==1), best_net.ACC);
        Ffit{gen+1} = Rfit;
        
        % (iv) Sort by rank
        [Rrank,idx] = sort(Rrank,'ascend');
        Rfit = Rfit(idx,:);
        R = R(idx,:);
        
        % (v) Compute the crowding distance index
        [Rcrowd, Rrank,~,R] = crowdingDistances(Rrank, Rfit, R);
        
        % (vi) Select Parent
        P = selectParentByRankAndDistance(Rcrowd, Rrank, R);
        
        %Update sampling terms
        [m, sigma] = update_m_sigma(gen, R, Ffit, m, sigma);
        fprintf('\t m =  %f, %f\n', m)
        fprintf('\t sigma =  %f, %f\n', sigma)
        
        % (vii) Compute child
        Q = applyCrossoverAndMutation(P, pc, pm, n_R, h_R, m, sigma);
        %             results(gen+1).gen = gen; results(gen+1).best_net=best_net;
        %             results(gen+1).acc = best_net.ACC;
        acc_log(gen+1,1) = best_net.ACC;
        fprintf(fileID, '\t For tarData %s, best architecture at Gen %d: %s\n',...
             tarData{ca}, best_net.nh, gen);
        
        if gen>2 && acc_log(gen+1)==acc_log(gen)&& acc_log(gen) == acc_log(gen-1)&& acc_log(gen-1)==100
            termination = 1;
        else
            gen = gen+1;
        end
    end
    % save(['./logs_CWRU/GSn2nOut_T' int2str(ca) '_L' int2str(ca) '_400'], 'best_net', 'acc_log')
%     load(['./logs_CWRU/GSn2nOut_T' int2str(ca) '_L' int2str(ca) '_400'], 'best_net', 'acc_log')
    
    xt = Y.test_inputs';
    yt = Y.test_results';
    if min(yt)==0
        yt = yt+1;
    end
    
    [acc,~, hData, pred] = valNetSfC(xt, yt, best_net);
    acc_mat(ca) = acc*100;
    fprintf(fileID, 'Accuracy of best model for %s: %.2f\n', tarData{ca}, acc*100);
end
% save('./logs_PBU/NSGA_net2net_40.mat', 'acc_mat', 'hData')
fclose(fileID);
% %%
% rmpath ('./ReqFnNSGAII'); rmpath ('./ReqFnNSGAII/softmax')
% rmpath ./ReqFnNSGAII/minFunc/;


