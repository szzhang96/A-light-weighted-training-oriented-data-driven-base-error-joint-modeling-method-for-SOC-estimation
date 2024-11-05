%% TRANSFERABLE MODEL TRAINING AND TEST
% *************************************************************************
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen
% *  - Joint First Authors
% ** - Corresponding Author
%
% Sturcture of this code:
% a. loading the training data
% b. train the BASE model
% c. train the TRANSFER model
% d. test
% *************************************************************************

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
rng(randomSeed)
%% a. load the training data
load '.\splitedData_A1\conditionNames.mat'
baseCondition = conditionNames{1,1};
fname = strcat('.\splitedData_A1\',baseCondition,'.mat');
load(fname)
baseDataTmp = data;
batteryGroupTmp = fieldnames(baseDataTmp);
batteryGroupTmp(1:4, :)= [] ;
baseDataAll = baseDataTmp.(batteryGroupTmp{1,1});
baseMdlXTrain = [];
baseMdlYTrain = [];
baseMdlYTest  = [];
for cycleNum = 1:cycleCutTrain:baseDataAll.cycleLife-1
    [baseMdlXTrainTmp, baseMdlYTrainTmp, baseMdlYTestTmp,~,~] = dataPreprocess(baseDataAll, cycleNum, baseDataTmp.rate1, baseDataTmp.rate2,...
        baseDataTmp.socChange);
    baseMdlXTrain = [baseMdlXTrain, baseMdlXTrainTmp];
    baseMdlYTrain = [baseMdlYTrain, baseMdlYTrainTmp];
    baseMdlYTest  = [baseMdlYTest; baseMdlYTestTmp];
end
%% b. train the BASE model
mu = mean(baseMdlXTrain(1, :));
sig = std(baseMdlXTrain(1, :));
baseMdlXTrain(1,:)=(baseMdlXTrain(1,:)- mu)./ sig;
numFeatures = size(baseMdlXTrain,1);
numResponses = 1;

% set the network structure and training options
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(128)
    reluLayer
    dropoutLayer(0.2)
    lstmLayer(128)
    fullyConnectedLayer(numResponses)
    regressionLayer];

transferableBaseMdlOptions = trainingOptions('rmsprop',...
    'MaxEpochs',1e3,...
    'GradientThreshold',1,...
    'InitialLearnRate', 0.01,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',200,...
    'LearnRateDropFactor',0.5,...
    'Verbose',0 );
tStart = tic;
transferableBaseMdl = trainNetwork(baseMdlXTrain,baseMdlYTrain,layers,transferableBaseMdlOptions);
tEnd = toc(tStart);
% check the training result with its own data
baseMdlYPred = predict(transferableBaseMdl, baseMdlXTrain);

save('.\trainedModels_B1\transferableBaseMdl.mat',"transferableBaseMdl","baseMdlYPred","baseMdlXTrain","baseMdlYTrain","transferableBaseMdlOptions","mu","sig","tEnd")

%% c. train the transferable model

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load .\trainedModels_B1\transferableBaseMdl.mat
load '.\splitedData_A1\conditionNames.mat'

for condition = 2: numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    batteryName = batteryGroup{1, 1};
    currentBatteryData = conditionData.(batteryName);

    XTrain = [];YTrain = [];YTest = [];
    fprintf('%s\t\t--%s\n', currentCondition, batteryName)

    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum = 1:cycleCutTrain:currentBatteryData.cycleLife-1
        currentData = currentBatteryData.(strcat('cycle',num2str(cycleNum)));
        if numel(currentData.t) > 3000
            continue
        end
        [transferableXTrainTmp, transferableYTrainTmp,~,~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1,...
            conditionData.rate2, conditionData.socChange);
        XTrain = [XTrain, transferableXTrainTmp];
        YTrain = [YTrain, transferableYTrainTmp] ;
    end
    XTrain(1,:) =(XTrain(1,:)-mu)./sig;

    layers = [
        transferableBaseMdl.Layers];

    layers(5,1).InputWeightsLearnRateFactor = 0;
    layers(5,1).RecurrentWeightsLearnRateFactor = 0;
    layers(5,1).BiasLearnRateFactor = 0;
    layers(6,1).WeightLearnRateFactor = 0;
    layers(6,1).BiasLearnRateFactor = 0;
    transferableOtherMdlOptions = trainingOptions('adam',...
        'MaxEpochs',1e3,...
        'GradientThreshold',1,...
        'InitialLearnRate', 0.0025,...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',200,...
        'LearnRateDropFactor',0.5,...
        'Verbose',0,...
        'Shuffle','every-epoch');
    tStart = tic;
    transferableOtherMdl = trainNetwork(XTrain, YTrain, layers, transferableOtherMdlOptions);
    tEnd = toc(tStart);
    transferableConditionMdlYPred = predict(transferableOtherMdl, XTrain);
    transferableConditionResult{condition, 1}{1,1} = 'Trained Transferable Model';
    transferableConditionResult{condition, 1}{1,2} = transferableOtherMdl;
    transferableConditionResult{condition, 1}{2,1} = 'Objective XTrain';
    transferableConditionResult{condition, 1}{2,2} = XTrain;
    transferableConditionResult{condition, 1}{3,1} = "Objective YTrain";
    transferableConditionResult{condition, 1}{3,2} = YTrain;
    transferableConditionResult{condition, 1}{4,1} = 'Objective YEst';
    transferableConditionResult{condition, 1}{4,2} = transferableConditionMdlYPred;
    transferableConditionResult{condition, 1}{5,1} = 'Transfer Training Time';
    transferableConditionResult{condition, 1}{5,2} = tEnd;
    transferableConditionResult{condition, 1}{6,1} = 'MAE';
    transferableConditionResult{condition, 1}{6,2} = mae(YTrain, transferableConditionMdlYPred);
    transferableConditionResult{condition, 1}{7,1} = 'RMSE';
    transferableConditionResult{condition, 1}{7,2} = rms((YTrain-transferableConditionMdlYPred));
end

save('.\results_B1\transferableConditionResult.mat',"transferableConditionResult")

%% d. test
clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load .\trainedModels_B1\transferableBaseMdl.mat
load .\splitedData_A1\conditionNames.mat
load .\results_B1\transferableConditionResult.mat

tic
for condition = 1: numel(conditionNames)
    currentCondition = conditionNames{condition, 1};
    fname = strcat('.\splitedData_A1\' ,currentCondition,'.mat');
    load(fname)
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup (1:4, :) = [];
    fprintf('%s\t\t--', currentCondition)
    if condition ~= 1
        currentMdl = transferableConditionResult{condition, 1}{1,2};
    else
        currentMdl = transferableBaseMdl;
    end
    for battery = 1: numel(batteryGroup)
        tic
        batteryName = batteryGroup{battery, 1};

        currentMdl = resetState(currentMdl);
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

            currentData = currentBatteryData.(cycleName) ;
            if numel(currentData.t) > 3000
                cyc = cyc+1;
                continue
            end
            currentMdl = resetState(currentMdl);

            [~,~, YRefTmp, ~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1, conditionData.rate2,...
                conditionData.socChange);
            k = 1;
            P = 1;
            resultWithoutKF = [];resultWithKF = [];
            while k < size(currentData,1)+1
                dataInputBase                = [(currentData.V(k)-mu)/sig, conditionData.rate1, conditionData.rate2,...
                    conditionData.socChange];

                [currentMdl, estResultSingleStep] = predictAndUpdateState(currentMdl, dataInputBase');
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

            result_transferableTest.(currentCondition).(batteryName).(cycleName).result_realQmax = [currentData.t, YRefTmp,resultWithoutKF(1,:)',resultWithKF(1,:)' ];
            maeKF_realQmax     = [maeKF_realQmax;    cyc, mean(abs(  YRefTmp-resultWithKF(1,:)'       ))];
            maeNoKF_realQmax   = [maeNoKF_realQmax;  cyc, mean(abs(  YRefTmp-resultWithoutKF(1,:)'    ))];
            rmseKF_realQmax    = [rmseKF_realQmax;   cyc, sqrt(mean((YRefTmp-resultWithKF(1,:)').^2   ))];
            rmseNoKF_realQmax  = [rmseNoKF_realQmax; cyc, sqrt(mean((YRefTmp-resultWithoutKF(1,:)').^2))];

            result_transferableTest.(currentCondition).(batteryName).(cycleName).result_fakeQmax = [currentData.t, YRefTmp,resultWithoutKF(2,:)',resultWithKF(2,:)' ];
            maeKF_fakeQmax     = [maeKF_fakeQmax;    cyc, mean(abs(  YRefTmp-resultWithKF(2,:)'       ))];
            maeNoKF_fakeQmax   = [maeNoKF_fakeQmax;  cyc, mean(abs(  YRefTmp-resultWithoutKF(2,:)'    ))];
            rmseKF_fakeQmax    = [rmseKF_fakeQmax;   cyc, sqrt(mean((YRefTmp-resultWithKF(2,:)').^2   ))];
            rmseNoKF_fakeQmax  = [rmseNoKF_fakeQmax; cyc, sqrt(mean((YRefTmp-resultWithoutKF(2,:)').^2))];

            if coCap > 1
                co = -1;
            else
                co = 1;
            end
            Qmax = [Qmax;cyc, currentData.Qc(end), currentData.Qc(end) *coCap, co*coCap];
            cyc = cyc+1;
        end
        result_transferableTest.(currentCondition).(batteryName).maeKF_realQmax  = maeKF_realQmax;
        result_transferableTest.(currentCondition).(batteryName).maeNoKF_realQmax  = maeNoKF_realQmax;
        result_transferableTest.(currentCondition).(batteryName).rmseKF_realQmax  = rmseKF_realQmax;
        result_transferableTest.(currentCondition).(batteryName).rmseNoKF_realQmax  = rmseNoKF_realQmax;

        result_transferableTest.(currentCondition).(batteryName).maeKF_fakeQmax  = maeKF_fakeQmax;
        result_transferableTest.(currentCondition).(batteryName).maeNoKF_fakeQmax  = maeNoKF_fakeQmax;
        result_transferableTest.(currentCondition).(batteryName).rmseKF_fakeQmax  = rmseKF_fakeQmax;
        result_transferableTest.(currentCondition).(batteryName).rmseNoKF_fakeQmax  = rmseNoKF_fakeQmax;
        result_transferableTest.(currentCondition).(batteryName).Qmax = Qmax;

        result_transferableTest.(currentCondition).(batteryName).average_maeKF_realQmax = mean(maeKF_realQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_rmseKF_realQmax = mean(rmseKF_realQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_maeNoKF_realQmax = mean(maeNoKF_realQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_rmseNoKF_realQmax = mean(rmseNoKF_realQmax(:,2));

        result_transferableTest.(currentCondition).(batteryName).average_maeKF_fakeQmax = mean(maeKF_fakeQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_rmseKF_fakeQmax = mean(rmseKF_fakeQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_maeNoKF_fakeQmax = mean(maeNoKF_fakeQmax(:,2));
        result_transferableTest.(currentCondition).(batteryName).average_rmseNoKF_fakeQmax = mean(rmseNoKF_fakeQmax(:,2));
        fname = strcat('.\results_B1\result_transferableTest_',currentCondition,'_',batteryName,'.mat');
        save(fname, "result_transferableTest")
        clear result_transferableTest
        toc
    end
end
toc
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
