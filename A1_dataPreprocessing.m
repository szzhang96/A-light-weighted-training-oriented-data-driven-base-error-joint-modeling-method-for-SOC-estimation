%% DATA PREPROCESSING
% *************************************************************************
% @CAO Ganglin*, @CHEN Shouxuan*, @GENG Yuanfei, @ZHANG Shuzhi**, @JIA Yao,
% @FENG Rong1, @ZHANG Xiongwen 
% *  - Joint First Authors
% ** - Corresponding Author
% 
% preprocessing the original data, including the following sub-parts:
% a. loading data
% b. split data 
% c. function unit
% *************************************************************************

clearvars -except Q R randomSeed cycleCutTrain cycleCutTest testFlag
%% a. loading data
warning off
load 2018-04-12_batchdata_updated_struct_errorcorrect.mat
mkdir splitedData_A1

%% b.sort and split data by protocol 
clearvars -except batch batch date Q R randomSeed cycleCutTrain cycleCutTest testFlag
for num = 1:size(batch,2)
    batch(num).batIdx = num;
end

[~,index] = sortrows({batch.policy}.');
batch = batch(index);

cyclecut = 1;
% pre defined some variables
rate1 = zeros(size(batch, 2),1);
rate2 = zeros(size(batch, 2), 1);
socChange = zeros(size(batch, 2), 1);
for batteryNum = 1: size(batch, 2)
    rateLoc = strfind(batch(batteryNum).policy, 'C_');
    socLoc = strfind(batch(batteryNum).policy,'PER');
    Cidx1 = batch(batteryNum).policy(1: rateLoc(1));
    Cidx2 = batch(batteryNum).policy(socLoc(1)+4:rateLoc(2));
    rate1(batteryNum) = func_getRate(Cidx1);
    rate2(batteryNum) = func_getRate(Cidx2);
    socChange(batteryNum)= str2double(batch(batteryNum).policy(socLoc(1)-2: socLoc(1)-1));
end
% find the unique protocol as reference
conditionFake = rate1+rate2+socChange;
conditionOnly = unique(conditionFake,"rows","stable");
conditionNames = cell(numel(conditionOnly),1);
for conditions = 1: numel(conditionOnly)
    groupTmp = find(conditionFake == conditionOnly(conditions));
    conditionName = strcat('condition_',num2str(rate1(groupTmp(1))), 'C_',num2str(socChange(groupTmp(1))),'_',...
        num2str(rate2(groupTmp(1))),'C');
    conditionNames{conditions,1} = conditionName;
    clear data
    data.conditionName = conditionName;
    data.rate1 = rate1(groupTmp(1))/10;
    data.rate2 = rate2(groupTmp(1))/10;
    data.socChange = socChange(groupTmp(1));

    % read the single battery data one by one
    for batteryNumber = groupTmp'
        realBatNum = batch(batteryNumber).batIdx;
        dataTmp = batch(batteryNumber).cycles;
        cycle_life = batch(batteryNumber).cycle_life;
        batName = strcat("batNum", num2str(realBatNum));
        cycleNum = size(batch(batteryNumber).cycles, 2);
        cycleGroup = 1: cyclecut: cycleNum;
        data.(batName).cycleLife = cycle_life;
        for cyc = cycleGroup
            if cyc > size(batch(batteryNumber).cycles, 2)
                continue
            end
            cycleName= strcat("cycle", num2str(cyc));

            charDis_flag_tmp = find(dataTmp(cyc).I < 0);
            charDis_flag_diff_tmp = diff(charDis_flag_tmp);
            charDis_flag_2tmp = find(charDis_flag_diff_tmp == 1,1);
            flagCharDis = charDis_flag_tmp(charDis_flag_2tmp)-1;
            % only need the charging stage
            t_real = 60*dataTmp(cyc).t(1: flagCharDis) ;
            V_real = dataTmp(cyc).V(1:flagCharDis);
            I_real = dataTmp(cyc).I(1:flagCharDis);
            Q_real = dataTmp(cyc).Qc(1: flagCharDis);
            V_ftResult = fit(t_real,V_real,'linearinterp','normalize','on');
            I_ftResult = fit(t_real,I_real,'linearinterp','normalize','on');
            Q_ftResult = fit(t_real,Q_real,'linearinterp','normalize','on');
            t_interp = 0:1:floor(t_real(end));
            V_interp = V_ftResult(t_interp);
            I_interp = I_ftResult(t_interp);
            Q_interp = Q_ftResult(t_interp);
            dataTableTmp = table(t_interp',V_interp, I_interp, ...
            Q_interp, 'VariableNames',{'t','V','I','Qc'});

            data.(batName).(cycleName)=dataTableTmp;
        end
    end
    fname = strcat('.\splitedData_A1\',conditionName,'.mat');
    fprintf('Now saving %s\t',fname)
    save(fname, "data")
    fprintf('\tsave complete\n')
end
% save the splited data
save('.\splitedData_A1\conditionNames.mat',"conditionNames")
%% c. functions unit
function rate = func_getRate(Cidx)
if ~isnan(find(Cidx=='_'))
    rate = str2double(Cidx(1))*10+str2double(Cidx(3));
else
    rate = str2double(Cidx(1))*10;
end 
end
