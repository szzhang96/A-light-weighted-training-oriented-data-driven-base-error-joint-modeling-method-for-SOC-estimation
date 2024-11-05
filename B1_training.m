%% TRAINING
% *************************************************************************
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen
% *  - Joint First Authors
% ** - Corresponding Author
%
% Sturcture of this code:
% a. load the first battery as base traing data (1st means sorted through
% A1)
% b. train the data to get the BASE model
% c. get the training data for ERROR model --> test the other conditions'
% first battery with well-trained BASE model
% d. get the final SOC estimation result --> combined with KF
% *************************************************************************

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
rng(randomSeed)

%% a.load the training data
load '.\splitedData_A1\conditionNames.mat'
baseCondition = conditionNames{1,1};
fname = strcat('.\splitedData_A1\',baseCondition,'.mat');
load(fname)
baseDataTmp = data;
batteryGroupTmp = fieldnames(baseDataTmp);
batteryGroupTmp(1:4, :)= [] ;
baseDataAll = baseDataTmp.(batteryGroupTmp{1,1});
[baseMdlXTrain, baseMdlYTrain, baseMdlYTest,~,~] = dataPreprocess(baseDataAll, 1, baseDataTmp.rate1, baseDataTmp.rate2,...
    baseDataTmp.socChange);

%% b.train the base model
mu = mean(baseMdlXTrain(1, :));
sig = std(baseMdlXTrain(1, :));
baseMdlXTrain(1,:)=(baseMdlXTrain(1,:)- mu)./ sig;
numFeatures = size(baseMdlXTrain,1);
numResponses = 1;

% set the network structure and training options
layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(10)
    lstmLayer(10)
    fullyConnectedLayer(numResponses)
    regressionLayer];

baseMdlOptions = trainingOptions('sgdm',...    
    'MaxEpochs',1.5e3,...
    'GradientThreshold',1,...
    'InitialLearnRate', 0.1,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',100,...
    'LearnRateDropFactor',0.7,...
    'Verbose',0 );
tStart = tic;
baseMdl = trainNetwork(baseMdlXTrain,baseMdlYTrain,layers,baseMdlOptions);
tEnd = toc(tStart);
% check the training result with its own data
baseMdlYPred = predict(baseMdl, baseMdlXTrain);
errBase = mae(baseMdlYPred, baseMdlYTrain);
fprintf('Base model trained, use %s seconds, the MAE is: %s\n',tEnd, errBase)

mkdir trainedModels_B1
save('.\trainedModels_B1\baseMdl.mat',"baseMdl","baseMdlYPred","baseMdlXTrain","baseMdlYTrain","baseMdlOptions","mu","sig","tEnd")
%% c. get the training data for ERROR model

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load '.\splitedData_A1\conditionNames.mat'
load '.\trainedModels_B1\baseMdl.mat'

tic
for condition = 1: numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    currentBatteryData = conditionData.(batteryGroup{1, 1});

    errMdlXTrainSingleBattery = [];errMdlYTrainSingleBattery = [];errMdlYTest = [];

    fprintf('%s\t\t--%s\n', currentCondition, batteryGroup{1,1})
    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum = 1:cycleCutTrain:currentBatteryData.cycleLife-1
        currentData = currentBatteryData.(strcat('cycle',num2str(cycleNum)));
        [XTrainTmp, YReal,~,Qc,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1,...
            conditionData.rate2, conditionData.socChange);
        XTrainTmp(1,:) =(XTrainTmp(1,:)-mu)./sig;
        estResultSingleCycle = predict(baseMdl, XTrainTmp) ;
        errMdlXTrainSingleBattery = [errMdlXTrainSingleBattery; estResultSingleCycle', XTrainTmp',...
            Qc(end) *ones(numel(estResultSingleCycle),1)];
        errMdlYTrainSingleBattery = [errMdlYTrainSingleBattery; YReal'] ;
    end
    errXTrainAllCondition{condition,1} = errMdlXTrainSingleBattery;
    errXTrainAllCondition{condition,2} = currentCondition;
    errYTrainAllCondition{condition,1} = errMdlYTrainSingleBattery;
end

errMdlXTrain = []; YReal = [] ;
for condition = 1:numel(conditionNames)
    errMdlXTrain = [errMdlXTrain; errXTrainAllCondition{condition,1}];
    YReal  = [YReal; errYTrainAllCondition{condition, 1}];
end
errMdlXTrain(:,2) = []; % delete the voltage
errMdlYTrain = errMdlXTrain(:,1)-YReal;
rng(1)
tStart = tic;
errMdl = fitrtree(errMdlXTrain, errMdlYTrain);% ardexponential got the best score
tEnd = toc(tStart);
errMdlTrainResult = predict(errMdl,errMdlXTrain);

save('.\trainedModels_B1\errMdl.mat', 'errMdl')
mkdir results_B1
save('.\results_B1\errMdlTrainResult.mat', "errMdlXTrain", "YReal","errMdlYTrain", "errXTrainAllCondition",...
    "errYTrainAllCondition", 'errMdlTrainResult',"tEnd")
%% d. get the final SOC estimation result
clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
load '.\splitedData_A1\conditionNames.mat'
load '.\trainedModels_B1\baseMdl.mat'
load '.\trainedModels_B1\errMdl.mat'

for condition = 1: numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData) ;
    batteryGroup(1:4, :) = [];
    batteryName = batteryGroup{1,1};
    currentBatteryData = conditionData.(batteryName);

    estResultSingleBatteryWithKF = [];estResultSingleBatteryWithoutKF = [];YRef = [];
    Qmax     = [];
    maeKF    = [];
    maeNoKF  = [];
    rmseKF   = [];
    rmseNoKF = [];
    cyc = 1;
    fprintf('%s\t\t--%s\n', currentCondition, batteryName)

    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum = 1:cycleCutTrain:currentBatteryData.cycleLife-1
        cycleName = strcat('cycle',num2str(cycleNum));
        currentData = currentBatteryData.(cycleName);
        [xTrainTmp,~, yRefTmp,~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1,...
            conditionData.rate2, conditionData.socChange);
        xTrainTmp(1,:) =(xTrainTmp(1,:)-mu)./sig;
        baseEstResultSingleCycle = predict(baseMdl, xTrainTmp) ;
        dataInputErr = [baseEstResultSingleCycle' ,ones(numel(baseEstResultSingleCycle),1)*conditionData.rate1,...
            ones(numel(baseEstResultSingleCycle),1)*conditionData.rate2,...
            ones(numel(baseEstResultSingleCycle),1)*conditionData.socChange,...
            ones(numel(baseEstResultSingleCycle), 1)*currentData.Qc(end) ];
        errEstResultSingleCycle = predict(errMdl,dataInputErr);
        k = 1;
        P = 1;
        resultWithKF = [];resultWithoutKF = [];
        baseMdl = resetState(baseMdl);
        while k < size(currentData, 1)+1
            estResultSingleCycle = baseEstResultSingleCycle(k)-errEstResultSingleCycle(k);
            resultWithoutKF(k) = estResultSingleCycle ;
            resultWithKF(k) = estResultSingleCycle;
            if k < 2
                socAh = 0;
                k = k+1;
                continue
            end
            socAh = resultWithKF(k-1)+100*(currentData.Qc(k)-currentData.Qc(k-1))/currentData.Qc(end) ;

            P1 = P+Q;
            K = P1/(P1+R);
            resultWithKF(k) = K*(estResultSingleCycle-socAh)+socAh;
            P = (1-K)*P1;
            k = k+1;
        end
        result_trainTest.(currentCondition).(batteryName).(cycleName).result = [currentData.t, yRefTmp,resultWithoutKF',resultWithKF' ];
        maeKF  = [maeKF; cyc, mean(abs(yRefTmp-resultWithKF'))];
        maeNoKF  = [maeNoKF; cyc, mean(abs(yRefTmp-resultWithoutKF'))];
        rmseKF  = [rmseKF; cyc, sqrt(mean((yRefTmp-resultWithKF').^2))];
        rmseNoKF  = [rmseNoKF; cyc, sqrt(mean((yRefTmp-resultWithoutKF').^2))];

        estResultSingleBatteryWithKF = [estResultSingleBatteryWithKF;resultWithKF'];
        estResultSingleBatteryWithoutKF = [estResultSingleBatteryWithoutKF;resultWithoutKF'];
        YRef = [YRef; yRefTmp ] ;
        Qmax = [Qmax;cyc, currentData.Qc(end)];
        cyc = cyc+1;
    end

    result_trainTest.(currentCondition).(batteryName).maeKF    = maeKF;
    result_trainTest.(currentCondition).(batteryName).maeNoKF  = maeNoKF;
    result_trainTest.(currentCondition).(batteryName).rmseKF   = rmseKF;
    result_trainTest.(currentCondition).(batteryName).rmseNoKF = rmseNoKF;
    result_trainTest.(currentCondition).(batteryName).result   = [YRef, estResultSingleBatteryWithoutKF, estResultSingleBatteryWithKF];
    result_trainTest.(currentCondition).(batteryName).Qmax     = Qmax;
end

mkdir results_B1
save('.\results_B1\trainingsetResult.mat',"result_trainTest")
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
YTest  = soc;
end