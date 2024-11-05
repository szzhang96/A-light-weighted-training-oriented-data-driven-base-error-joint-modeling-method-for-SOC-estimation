%% *MAIN* 
% *************************************************************************
% This code is the main part of the series code files of --Towards accurate
% state-of-charge online estimation during various multi-stage constant
% current fast-charging protocols over battery entire lifespan: A
% light-weighted-training oriented data-driven base-error joint modeling
% method--, it sets the initial value of Q, R and randomSeed for the
% training and test.
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen 
% *  - Joint First Authors
% ** - Corresponding Author
% *************************************************************************

clc
clear
Q = 1e-6;
R = 1e-2;
randomSeed = 1;
cycleCutTrain = 100;
cycleCutTest = 1;
testFlag = 1;

%% data preprocessing
tic
if testFlag == 0
    disp('start data preprocessing')
    run A1_dataPreprocessing.m
    disp('preprocessing done')
end
toc

%% Proposed method
    %% training
    tic
    disp('start training')
    run B1_training.m
    disp('training complete')
    toc
    
    %% test
    tic
    disp('start testing')
    run C1_test.m
    disp('test complete')
    toc

%% transferable model
tic
disp('start transferable model comparison')
run D1_transferableModelTest.m
disp('transferable model over')
toc

%% unified model
tic
disp('start unified model comparison')
run E1_unifiedModelTest.m
disp('unified model over')
toc

