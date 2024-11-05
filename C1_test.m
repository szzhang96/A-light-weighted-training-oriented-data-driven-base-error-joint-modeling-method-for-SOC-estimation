%% TEST
% *************************************************************************
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen
% *  - Joint First Authors
% ** - Corresponding Author
%
% Sturcture of this code:
% a. load the test data
% b. test --> test through BASE and ERROR model, then combined with KF to get the
%    final SOC estimation result
% *************************************************************************

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag

%% a.load the test data
load '.\splitedData_A1\conditionNames.mat'
load '.\trainedModels_B1\baseMdl.mat'
load '.\trainedModels_B1\errMdl.mat'

for condition = 1:numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    %% b. test
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    fprintf('\n%s\t\t--', currentCondition)
    for battery = 1: numel(batteryGroup)
        batteryName = batteryGroup{battery, 1};
        fprintf('%s\n\t\t', batteryName)
        currentBatteryData = conditionData.(batteryName);
        YRef = [];
        estResultSingleBatteryWithKF = [];estResultSingleBatteryWithoutKF = [];

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
            if numel(currentData.t) > 4000
                cyc = cyc+1;
                continue
            end
            [~,~, YRefTmp, ~,~] = dataPreprocess(currentBatteryData, cycleNum, conditionData.rate1, conditionData.rate2,...
                conditionData.socChange);
            k = 1;
            P = 1;
            resultWithoutKF = [];resultWithKF = [];
            baseMdl = resetState(baseMdl);
            while k < size(currentData,1)+1
                dataInputBase                = [(currentData.V(k)-mu)/sig, conditionData.rate1, conditionData.rate2,...
                    conditionData.socChange];

                [baseMdl, baseEstResultSingleStep] = predictAndUpdateState(baseMdl, dataInputBase');
                nnum = 1;
                for coCap = [1  1+(4*rand-2)/100]
                    dataInputErr                    = [baseEstResultSingleStep, conditionData.rate1, conditionData.rate2,...
                        conditionData.socChange, currentData.Qc(end) *coCap];
                    errEstResultSingleStep = predict(errMdl, dataInputErr);
                    estResultSingleStep = baseEstResultSingleStep-errEstResultSingleStep;
                    resultWithoutKF(nnum, k) = estResultSingleStep;
                    resultWithKF(nnum, k) = estResultSingleStep;
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
            result_test.(currentCondition).(batteryName).(cycleName).result_realQmax = [currentData.t, YRefTmp,resultWithoutKF(1,:)',resultWithKF(1,:)' ];
            maeKF_realQmax     = [maeKF_realQmax;    cyc, mean(abs(  YRefTmp-resultWithKF(1,:)'       ))];
            maeNoKF_realQmax   = [maeNoKF_realQmax;  cyc, mean(abs(  YRefTmp-resultWithoutKF(1,:)'    ))];
            rmseKF_realQmax    = [rmseKF_realQmax;   cyc, sqrt(mean((YRefTmp-resultWithKF(1,:)').^2   ))];
            rmseNoKF_realQmax  = [rmseNoKF_realQmax; cyc, sqrt(mean((YRefTmp-resultWithoutKF(1,:)').^2))];

            result_test.(currentCondition).(batteryName).(cycleName).result_fakeQmax = [currentData.t, YRefTmp,resultWithoutKF(2,:)',resultWithKF(2,:)' ];
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
        result_test.(currentCondition).(batteryName).maeKF_realQmax  = maeKF_realQmax;
        result_test.(currentCondition).(batteryName).maeNoKF_realQmax  = maeNoKF_realQmax;
        result_test.(currentCondition).(batteryName).rmseKF_realQmax  = rmseKF_realQmax;
        result_test.(currentCondition).(batteryName).rmseNoKF_realQmax  = rmseNoKF_realQmax;

        result_test.(currentCondition).(batteryName).maeKF_fakeQmax  = maeKF_fakeQmax;
        result_test.(currentCondition).(batteryName).maeNoKF_fakeQmax  = maeNoKF_fakeQmax;
        result_test.(currentCondition).(batteryName).rmseKF_fakeQmax  = rmseKF_fakeQmax;
        result_test.(currentCondition).(batteryName).rmseNoKF_fakeQmax  = rmseNoKF_fakeQmax;
        result_test.(currentCondition).(batteryName).Qmax = Qmax;

        result_test.(currentCondition).(batteryName).average_maeKF_realQmax = mean(maeKF_realQmax(:,2));
        result_test.(currentCondition).(batteryName).average_rmseKF_realQmax = mean(rmseKF_realQmax(:,2));
        result_test.(currentCondition).(batteryName).average_maeNoKF_realQmax = mean(maeNoKF_realQmax(:,2));
        result_test.(currentCondition).(batteryName).average_rmseNoKF_realQmax = mean(rmseNoKF_realQmax(:,2));

        result_test.(currentCondition).(batteryName).average_maeKF_fakeQmax = mean(maeKF_fakeQmax(:,2));
        result_test.(currentCondition).(batteryName).average_rmseKF_fakeQmax = mean(rmseKF_fakeQmax(:,2));
        result_test.(currentCondition).(batteryName).average_maeNoKF_fakeQmax = mean(maeNoKF_fakeQmax(:,2));
        result_test.(currentCondition).(batteryName).average_rmseNoKF_fakeQmax = mean(rmseNoKF_fakeQmax(:,2));
        fname = strcat('.\results_B1\test_',currentCondition,'_',batteryName,'.mat');
        save(fname, "result_test")
        clear result_test
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