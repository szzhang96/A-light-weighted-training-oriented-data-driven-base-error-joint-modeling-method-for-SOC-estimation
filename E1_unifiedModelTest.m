%% For unified model test
% *************************************************************************
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen 
% *  - Joint First Authors
% ** - Corresponding Author
% 
% Sturcture of this code:
% a. loading the training data
% b. train the unified model
% c. test

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
rng(randomSeed)
%% a. load the training data
load '.\splitedData_A1\conditionNames.mat'

unifiedXTrain = [];
unifiedYTrain = [];
for condition = 1: numel(conditionNames)
    conditionTmp = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',conditionTmp,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    batteryName = batteryGroup{1, 1};
    currentBatteryData = conditionData.(batteryName);
    fprintf('%s\t\t--%s\n', conditionTmp, batteryName)
    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum = 1:cycleCutTrain:currentBatteryData.cycleLife-1
        currentData = currentBatteryData.(strcat('cycle',num2str(cycleNum)));
        if numel(currentData.t) > 3000
            continue
        end
        [unifiedXTrainTmp, unifiedYTrainTmp,~,~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1,...
            conditionData.rate2, conditionData.socChange);
        unifiedXTrain = [unifiedXTrain, unifiedXTrainTmp];
        unifiedYTrain = [unifiedYTrain, unifiedYTrainTmp] ;
    end
end
mu = mean(unifiedXTrain(1, :));
sig = std(unifiedXTrain(1, :));
unifiedXTrain(1,:) =(unifiedXTrain(1,:)-mu)./sig;
unifiedYTrain = unifiedYTrain';

numFeatures = size(unifiedXTrain,1);
numResponses = 1;
%% b. train the unified model

% set the network structure and training options
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(256)
    reluLayer
    dropoutLayer(0.2)
    lstmLayer(128)
    fullyConnectedLayer(numResponses)
    regressionLayer];

unifiedMdlOptions = trainingOptions('rmsprop',...
    'MaxEpochs',1e3,...
    'GradientThreshold',1,...
    'InitialLearnRate', 0.01,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',200,...
    'LearnRateDropFactor',0.5,...
    'Verbose',0 );
tStart = tic;
unifiedMdl = trainNetwork(unifiedXTrain,unifiedYTrain',layers,unifiedMdlOptions);
tEnd = toc(tStart);
% check the training result with its own data
unifiedYPred = predict(unifiedMdl, unifiedXTrain);

save('.\trainedModels_B1\unifiedMdl.mat',"unifiedMdl","unifiedYPred","unifiedXTrain","unifiedYTrain","unifiedMdlOptions","mu","sig","tEnd")

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load .\trainedModels_B1\unifiedMdl.mat
load '.\splitedData_A1\conditionNames.mat'

for condition = 1: numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    batteryName = batteryGroup{1, 1};
    currentBatteryData = conditionData.(batteryName);
    fprintf('%s\t\t--%s\n', currentCondition, batteryName)
    unifiedXTrain = [];unifiedYTrain = [];
    YRef = [];
    Qmax= [];
    maeKF  = [];
    maeNoKF  = [];
    rmseKF  = [];
    rmseNoKF = [];
    estResultSingleBatteryWithKF=[];
    estResultSingleBatteryWithoutKF =[ ];
    oneYTest = [];
    cyc = 1;
    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum = 1:cycleCutTrain:currentBatteryData.cycleLife-1
        cycleName = strcat('cycle',num2str(cycleNum));
        currentData = currentBatteryData.(cycleName);
        if numel(currentData.t) > 3000
            cyc = cyc+1;
            continue
        end
        [unifiedXTrainTmp, ~,YRefTmp,~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1,...
            conditionData.rate2, conditionData.socChange);
        unifiedXTrainTmp(1,:) = (unifiedXTrainTmp(1,:)-mu)./sig;
        [unifiedMdl, estResultSingleCycle] = predictAndUpdateState(unifiedMdl, unifiedXTrainTmp) ;

        k = 1;
        P = 1;
        resultWithKF = [];resultWithoutKF = [];

        while k < size(currentData, 1)+1
            resultWithoutKF(k) = estResultSingleCycle(k) ;
            resultWithKF(k) = estResultSingleCycle(k);
            if k < 2
                socAh = 0;
                k = k+1;
                continue
            end
            socAh = resultWithKF(k-1)+100*(currentData.Qc(k)-currentData.Qc(k-1))/currentData.Qc(end) ;

            P1 = P+Q;
            K = P1/(P1+R);
            resultWithKF(k) = K*(estResultSingleCycle(k)-socAh)+socAh;
            P = (1-K)*P1;
            k = k+1;
        end

        result_unifiedTrain.(currentCondition).(batteryName).(cycleName).result = [currentData.t, YRefTmp,resultWithoutKF',resultWithKF' ];
        maeKF  = [maeKF; cyc, mean(abs(YRefTmp-resultWithKF'))];
        maeNoKF  = [maeNoKF; cyc, mean(abs(YRefTmp-resultWithoutKF'))];
        rmseKF  = [rmseKF; cyc, sqrt(mean((YRefTmp-resultWithKF').^2))];
        rmseNoKF  = [rmseNoKF; cyc, sqrt(mean((YRefTmp-resultWithoutKF').^2))];

        estResultSingleBatteryWithKF = [estResultSingleBatteryWithKF;resultWithKF'];
        estResultSingleBatteryWithoutKF = [estResultSingleBatteryWithoutKF;resultWithoutKF'];
        oneYTest = [oneYTest; YRefTmp ] ;
        Qmax = [Qmax;cyc, currentData.Qc(end)];
        cyc = cyc+1;
    end
    result_unifiedTrain.(currentCondition).(batteryName).maeKF  = maeKF;
    result_unifiedTrain.(currentCondition).(batteryName).maeNoKF  = maeNoKF;
    result_unifiedTrain.(currentCondition).(batteryName).rmseKF  = rmseKF;
    result_unifiedTrain.(currentCondition).(batteryName).rmseNoKF  = rmseNoKF;
    result_unifiedTrain.(currentCondition).(batteryName).result = [YRef, estResultSingleBatteryWithoutKF, estResultSingleBatteryWithKF];
    result_unifiedTrain.(currentCondition).(batteryName).Qmax = Qmax;
end

save('.\results_B1\result_unifiedTrain.mat',"result_unifiedTrain")
%% c. test
clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load .\trainedModels_B1\unifiedMdl.mat
load .\splitedData_A1\conditionNames.mat

for condition = 1: numel(conditionNames)
    currentCondition = conditionNames{condition, 1};
    fname = strcat('.\splitedData_A1\' ,currentCondition,'.mat');
    load(fname)
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup (1:4, :) = [];
    fprintf('%s\t\t--', currentCondition)

    for battery = 1: numel(batteryGroup)
        tic
        batteryName = batteryGroup{battery, 1};
        currentBatteryData = conditionData.(batteryName);
        fprintf('%s\n\t\t', batteryName)
        estResultSingleBatteryWithKF = [];estResultSingleBatteryWithoutKF = [];
        YRef = [];
        Qmax= [];
        maeKF_realQmax  = [];
        maeNoKF_realQmax  = [];
        rmseKF_realQmax  = [];
        rmseNoKF_realQmax = [];
        maeKF_fakeQmax  = [];
        maeNoKF_fakeQmax  = [];
        rmseKF_fakeQmax  = [];
        rmseNoKF_fakeQmax = [];
        cyc = 1;

        if isnan(currentBatteryData.cycleLife)
            currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData)) -1);
        end
        for cycleNum = 1:cycleCutTest:currentBatteryData.cycleLife-1
            cycleName = strcat('cycle',num2str(cycleNum));
            currentData = currentBatteryData.(cycleName);
            if numel(currentData.t) > 3000
                cyc = cyc+1;
                continue
            end
            if cycleNum == 201
                disp('check')
            end
            unifiedMdl = resetState(unifiedMdl);
            currentData = currentBatteryData.(cycleName) ;
            [~,~, YRefTmp, ~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1, conditionData.rate2,...
                conditionData.socChange);
            k = 1;
            P = 1;
            resultWithoutKF = [];resultWithKF = [];
            while k < size(currentData,1)+1
                dataInputBase                = [(currentData.V(k)-mu)/sig, conditionData.rate1, conditionData.rate2,...
                    conditionData.socChange];

                [unifiedMdl, estResultSingleStep] = predictAndUpdateState(unifiedMdl, dataInputBase');

                nnum = 1;
                for coCap = [1  1+(4*rand-2)/100]
                    resultWithoutKF(nnum, k) = estResultSingleStep;
                    resultWithKF(nnum, k)                 = estResultSingleStep;
                    if k<2
                        resultWithKF(nnum, k) = resultWithoutKF(nnum, k) ;
                        k = k+1;
                        continue
                    end
                    socAh = resultWithKF(nnum, k-1)+100*(currentData.Qc(k) -currentData.Qc(k-1)) / currentData.Qc(end) /coCap;

                    P1 = P+Q;
                    K = P1/(P1+R);
                    resultWithKF(nnum, k) = K*(estResultSingleStep-socAh)+socAh;
                    P =(1-K)*P1;
                    nnum = nnum+1;
                end
                k = k+1;
            end

            result_unifiedTest.(currentCondition).(batteryName).(cycleName).result_realQmax = [currentData.t, YRefTmp,resultWithoutKF(1,:)',resultWithKF(1,:)' ];
            maeKF_realQmax     = [maeKF_realQmax;    cyc, mean(abs(  YRefTmp-resultWithKF(1,:)'       ))];
            maeNoKF_realQmax   = [maeNoKF_realQmax;  cyc, mean(abs(  YRefTmp-resultWithoutKF(1,:)'    ))];
            rmseKF_realQmax    = [rmseKF_realQmax;   cyc, sqrt(mean((YRefTmp-resultWithKF(1,:)').^2   ))];
            rmseNoKF_realQmax  = [rmseNoKF_realQmax; cyc, sqrt(mean((YRefTmp-resultWithoutKF(1,:)').^2))];

            result_unifiedTest.(currentCondition).(batteryName).(cycleName).result_fakeQmax = [currentData.t, YRefTmp,resultWithoutKF(2,:)',resultWithKF(2,:)' ];
            maeKF_fakeQmax     = [maeKF_fakeQmax;    cyc, mean(abs(  YRefTmp-resultWithKF(2,:)'       ))];
            maeNoKF_fakeQmax   = [maeNoKF_fakeQmax;  cyc, mean(abs(  YRefTmp-resultWithoutKF(2,:)'    ))];
            rmseKF_fakeQmax    = [rmseKF_fakeQmax;   cyc, sqrt(mean((YRefTmp-resultWithKF(2,:)').^2   ))];
            rmseNoKF_fakeQmax  = [rmseNoKF_fakeQmax; cyc, sqrt(mean((YRefTmp-resultWithoutKF(2,:)').^2))];

            YRef = [YRef; YRefTmp ];
            if coCap > 1
                co = -1;
            else
                co = 1;
            end
            Qmax = [Qmax;cyc, currentData.Qc(end), currentData.Qc(end) *coCap, co*coCap];
            cyc = cyc+1;
        end
        result_unifiedTest.(currentCondition).(batteryName).maeKF_realQmax  = maeKF_realQmax;
        result_unifiedTest.(currentCondition).(batteryName).maeNoKF_realQmax  = maeNoKF_realQmax;
        result_unifiedTest.(currentCondition).(batteryName).rmseKF_realQmax  = rmseKF_realQmax;
        result_unifiedTest.(currentCondition).(batteryName).rmseNoKF_realQmax  = rmseNoKF_realQmax;
        % result_unifiedTest.(currentCondition).(batteryName).result_realQmax = [YRef, estResultSingleBatteryWithoutKF(:,1), estResultSingleBatteryWithKF(:,1)];

        result_unifiedTest.(currentCondition).(batteryName).maeKF_fakeQmax  = maeKF_fakeQmax;
        result_unifiedTest.(currentCondition).(batteryName).maeNoKF_fakeQmax  = maeNoKF_fakeQmax;
        result_unifiedTest.(currentCondition).(batteryName).rmseKF_fakeQmax  = rmseKF_fakeQmax;
        result_unifiedTest.(currentCondition).(batteryName).rmseNoKF_fakeQmax  = rmseNoKF_fakeQmax;
        % result_unifiedTest.(currentCondition).(batteryName).result_fakeQmax = [YRef, estResultSingleBatteryWithoutKF(:,2), estResultSingleBatteryWithKF(:,2)];
        result_unifiedTest.(currentCondition).(batteryName).Qmax = Qmax;


        result_unifiedTest.(currentCondition).(batteryName).average_maeKF_realQmax = mean(maeKF_realQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_rmseKF_realQmax = mean(rmseKF_realQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_maeNoKF_realQmax = mean(maeNoKF_realQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_rmseNoKF_realQmax = mean(rmseNoKF_realQmax(:,2));

        result_unifiedTest.(currentCondition).(batteryName).average_maeKF_fakeQmax = mean(maeKF_fakeQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_rmseKF_fakeQmax = mean(rmseKF_fakeQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_maeNoKF_fakeQmax = mean(maeNoKF_fakeQmax(:,2));
        result_unifiedTest.(currentCondition).(batteryName).average_rmseNoKF_fakeQmax = mean(rmseNoKF_fakeQmax(:,2));

        fname = strcat('.\results_B1\result_unifiedTest_',currentCondition,'_',batteryName,'.mat');
        save(fname, "result_unifiedTest")
        clear result_unifiedTest
        toc
    end
end
%% function units
function [XTrain, YTrain, YTest, Qc, CurrentAndVoltage] = dataPreprocess(dataChoose, cycleNum, rate1, rate2, socChange)
if isnan(cycleNum)
    XTrain = [];
    YTrain = [];
    YTest = [];
    Qc = [];
    CurrentAndVoltage = [];
    return
end
charData = dataChoose.(strcat('cycle', num2str(cycleNum))) ;
sz = size(charData,1);
rateOne = rate1 * ones(sz,1);
socChange = socChange*ones(sz, 1) ;
rateTwo = rate2*ones(sz, 1) ;
Qc = charData.Qc;
soc = 100*charData.Qc/charData.Qc(end) ;
CurrentAndVoltage = [charData.V, charData.I];
XTrain = [charData.V, rateOne, rateTwo, socChange]';
YTrain = soc';
YTest = soc;
end
