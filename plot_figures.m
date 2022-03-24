%% Figure 1C & 3A(middle): PC as a function of trial iteration and condition (across participants)

clear all

% NOTE: toggle between two experiments for different plots
exptype = 'Mturk'; % for figure 1D
% exptype = 'RPP'; % for figure 4

load('experimentalsettings.mat')
nBlocks = 12;

subjcorrMat = cell(1,nConds);
for isubj = 1:nSubjs.(exptype) % for each participant
    subjid = subjidVec.(exptype)(isubj);
    
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    corrMat = cell(1,nConds);
    for iblock = 1:nBlocks % for each block
        
        nStims = nStimsVec(iblock); % number of stimuli in current block
        condition = condVec(iblock); % current image condition
        
        icond = find(conditionMat(1,:)==nStims & conditionMat(2,:)==condition);
        iscorrVec = corrCell{iblock};
        
        for istim = 1:nStims % for each stimulus
            corrvec = iscorrVec(stimvaluesCell{iblock} == istim); % all relevant stim times
            corrvec(corrvec == -1) = nan;
            
            corrMat{icond} = [corrMat{icond}; corrvec(1:nTimes_min)];
        end
    end
    
    for icond = 1:nConds % average across all stimuli in each condition
        %         corrMat{icond}(corrMat{icond} == -1) = 0;
        subjcorrMat{icond}(isubj,:) = nanmean(corrMat{icond});
    end
end

% plot learning curve for each of the conditions, averaging across stimuli
figure; hold on;
for icond = 1:nConds
    m = mean(subjcorrMat{icond});
    sem = std(subjcorrMat{icond})./sqrt(nSubjs.(exptype));
    errorbar(m,sem,'Color',colorMat(icond,:),'CapSize',14);
end
ylim([0 1])
xlim([0.5 11.5])
set(gca,'XTick',0:11)
xlabel('stimulus iteration')
ylabel('proportion correct')
legend({'Visual 3','Conjunctive 3','Linguistic 3','Visual 6','Conjunctive 6','Linguistic 6'})
defaultplot % changing various default plotting aesthetics


%% FIGURE 1D & 3A(right): logistic regression betas

clear all
load('experimentalsettings.mat')
exptype = 'RPP';

dev = nan(nSubjs.(exptype),3);
nTrialsVec = nan(nSubjs.(exptype),1);
for isubj = 1:nSubjs.(exptype)
    subjid = subjidVec.(exptype)(isubj);
    
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    for icond = 1:3 % picture conditions
        idx_block = find(condVec == conditionMat(2,icond));
        
        dataMat = [];
        for iblock = idx_block
            nStims = nStimsVec(iblock);
            
            for istim = 1:nStims
                idx = find(stimvaluesCell{iblock} == istim);
                corrVec = corrCell{iblock}(idx)';
                idx_first = find(corrVec == 1,1,'first');
                
                if sum(corrVec > 0)
                    corrVec = corrVec(idx_first:end);
                    idx = idx(idx_first:end);
                    %                     nCorrVec = cumsum(corrVec);
                    
                    % get delays since last correct
                    idx1 = idx(1);
                    delaycorrVec = nan(length(corrVec),1);
                    for ii = 2:length(corrVec)
                        delaycorrVec(ii) = idx(ii)-idx1;
                        if (corrVec(ii) == 1) % if got current correct
                            idx1 = idx(ii); % update the most recent correct
                        end
                    end
                    
                    dMat = [corrVec nStims*ones(length(corrVec),1) -1./delaycorrVec cumsum(corrVec)-corrVec];
                    dMat(1,:) = [];
                    dataMat = [dataMat; dMat];
                end
            end
        end
        idx_noresp = (dataMat(:,1) == -1);
        dataMat(idx_noresp,:) = [];
        
        % z scoring predictors
        dataMat(:,2:end) = (dataMat(:,2:end)-nanmean(dataMat(:,2:end)))./nanstd(dataMat(:,2:end));
        
        nTrialsVec(isubj) = size(dataMat,1);
        [B(:,isubj,icond), dev(isubj,icond)] = mnrfit(dataMat(:,2:end),categorical(2-dataMat(:,1)),'model','ordinal','Link','logit');
        
    end
end

nPreds = size(B,1);
AICcVec = bsxfun(@plus,-dev,2*nPreds+2*nPreds*(nPreds+1)./(nTrialsVec-nPreds-1));

plotorder = [4 2 3 1];
B = tanh(B);

figure;
plot([0.5,4.5],[0 0],'Color',0.7*ones(1,3))
for icond = 1:3
    
    betaMat = B(:,:,icond)';
    
    jitter = rand(nSubjs.(exptype),1)*.1;
    if (icond == 2)
        jitter=jitter-0.1;
    elseif (icond==1)
        jitter=jitter+0.1;
    end
    
    m = nanmean(betaMat);
    sem = nanstd(betaMat)./sqrt(nSubjs.(exptype));
    
    hold on;
    plot(bsxfun(@plus,1:nPreds,jitter),betaMat(:,plotorder),'.','Color',colorMat(icond+3,:))
    errorbar([1:nPreds]+jitter(1),m(plotorder),sem(plotorder),'Color','k','LineStyle','none')
    %         errorbar([1:nPreds]+jitter(1),m,sem,'Color','k','LineStyle','none')
    set(gca,'XTick',1:nPreds,'XTickLabel',{'n correct','set size','delay','intercept'})
end
ylabel('tanh(beta)')
xlim([0.5 nPreds+0.5])
defaultplot




%% Figure 3B(middle,right): IDENITY PLOT: TRAIN AGAINST TEST

clear all

load('experimentalsettings.mat')
exptype = 'RPP';
nSubjs = nSubjs.(exptype);
subjidVec = subjidVec.(exptype);

[data_learn_pcMat, data_test_pcMat] = deal(nan(nSubjs,nConds));
for isubj = 1:nSubjs
    subjid = subjidVec(isubj);
    
    % ----- load data ------
    % learning data
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    % ----- stuff for plotting -----
    [data_test_pc,data_learn_pc] = deal(nan(2,nConds));
    for icond = 1:nConds
        idx_imagecond = condVec == conditionMat(2,icond);
        idx_ns = cellfun(@max,stimvaluesCell,'UniformOutput',true) == conditionMat(1,icond);
        blocknumVec = find(idx_imagecond & idx_ns);
        nBlocks = length(blocknumVec);
        
        for iblock = 1:nBlocks
            blocknum = blocknumVec(iblock);
            
            idx_learn_useabletrials = corrCell{blocknum} ~= -1;
            idx_test_useabletrials = test_subjrespCell{blocknum} ~= -1;
            data_learn_pc(iblock,icond) = mean(corrCell{blocknum}(idx_learn_useabletrials));
            data_test_pc(iblock,icond) = mean(test_corrrespCell{blocknum}(idx_test_useabletrials) == test_subjrespCell{blocknum}(idx_test_useabletrials));
        end
    end
    
    data_learn_pcMat(isubj,:) = mean(data_learn_pc);
    data_test_pcMat(isubj,:) = mean(data_test_pc);
end

% identity plot
figure; hold on;
plot([0 1],[0 1],'-','Color',0.7*ones(1,3));
for icond = 1:nConds
    x = data_learn_pcMat(:,icond);
    y = data_test_pcMat(:,icond);
    plot(x,y,'.','Color',colorMat(icond,:),'MarkerSize',14)
    
    m_x = mean(x);
    sem_x = std(x)./sqrt(length(x));
    m_y = mean(y);
    sem_y = std(y)./sqrt(length(y));
    errorbar(m_x,m_y,sem_y,sem_y,sem_x,sem_x,'Color',colorMat(icond,:),...
        'CapSize',14)
end
xlabel('proportion correct learn')
ylabel('proportion correct test')
defaultplot


%% FIG 2, 4, 5(top): LEARNING PHASE MODEL VALIDATION

clear all

% ======= TOGGLE EXPERIMENT AND MODEL NAME =====

% EXPERIMENT NAME
exptype = 'RPP';  % experiment 1
% exptype = 'RPP';    % experiment 2

% % EXPERIMENT 1 MODELS
% model = 'RL3WM_pers_0';               % RL learning rate
% model = 'RLWM_CA_pers_0';           % RL credit assignment
% model = 'RLWM_dn_pers_sub_0';       % RL decision noise
% model = 'Decay3';                   % WM decay
% model = 'RLWM_dnwm_pers_sub_0';     % WM decision noise
% model = 'ns2_pers_0';               % RL-WM weight
% % EXPERIMENT 2 MODELS (in addition to those above)
% model = 'RL3WMi_pers_0';            % RL learning rate with interaction
% model = 'RLWMi_dn_pers_sub_0';      % RL decision noise with interaction
% model = 'RL3WMi_all_pers_0';        % RL learning rate with interaction (fitted on learn+test data)
% model = 'RLWMi_dn_all_pers_sub_0';  % RL decision noise with interaction (fitted on learn+test data)

% =======  SIMULATE DATA ======
if strcmp(model(end-1:end),'_0') % if model has negative learning rate of 0
    modell = model(1:end-2);
    if strcmp(modell(end-3:end),'_sub') % if dn or CA model not fitting a parameter for Category condition
        modell = modell(1:end-4);
    end
else
    modell = model;
end

try
    if strcmp(modell(end-8:end-5),'_all')
        modell = [modell(1:end-9) modell(end-4:end)];
    end
end

nSims = 10; % number of simulations per participant to average

% logicals for different plots
plot_datamodel_overlay = 1;     % overlaid in classic validation way
plot_datamodel_sep = 0;         % plot data and model fits separtely in two subplots
plot_indvlsubjects = 0;         % plot data for each participant
plot_indvlmodels = 0;           % plot model simulations for each participant

% experimental settings
load('experimentalsettings.mat')
nTimes = 11; % min how many times each stim is presented

% load MLE parameters
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model))

% load and add in fixed parameters from model
[~,~,~,~,~,~,~,~,~,~,fixparams] = loadfittingparams(model);
nps = size(xbestMat,2) + size(fixparams,2);
nv = 1:nps;
nv(fixparams(1,:)) = [];
xb(:,nv) = xbestMat;
xb(:,fixparams(1,:)) = repmat(fixparams(2,:),nSubjs.(exptype),1);
xbestMat = xb;

[modcorrMat, subjcorrMat] = deal(cell(1,nConds));
for isubj = 1:nSubjs.(exptype)
    subjid = subjidVec.(exptype)(isubj);
    
    % load data
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    % ------------------ simulating data ------------------
    modcorrCellCell = cell(1,nSims);
    for isim = 1:nSims
        eval(sprintf('[~, modcorrCellCell{isim}] = simulate_%s(xbestMat(isubj,:),stimvaluesCell,corrrespCell,condVec,responseCell);',modell))
    end
    
    % average across stimulations
    nBlocks = length(modcorrCellCell{1});
    modcorrCell = cell(1,nBlocks);
    for iblock = 1:nBlocks
        modcorrVec = [];
        for isim = 1:nSims
            modcorrVec = [modcorrVec; modcorrCellCell{isim}{iblock}];
        end
        modcorrCell{iblock} = nanmean(modcorrVec);
    end
    
    % -------------- breaking things up by condition -------------
    [modcorrmat,corrMat] = deal(cell(1,nConds));
    for iblock = 1:12
        nStims = nStimsVec(iblock);
        condition = condVec(iblock);
        icond = find(conditionMat(1,:)==nStims & conditionMat(2,:)==condition);
        
        for istim = 1:nStims
            corrvec = corrCell{iblock}(stimvaluesCell{iblock} == istim); % all relevant stim times
            modelcorrvec = modcorrCell{iblock}(stimvaluesCell{iblock} == istim);
            
            corrMat{icond} = [corrMat{icond}; corrvec(1:nTimes)];
            modcorrmat{icond} = [modcorrmat{icond}; modelcorrvec(1:nTimes)];
        end
    end
    
    for icond = 1:nConds
        corrMat{icond}(corrMat{icond} == -1) = nan;
        modcorrmat{icond}(modcorrmat{icond} == -1) = nan;
        subjcorrMat{icond}(isubj,:) = nanmean(corrMat{icond});
        modcorrMat{icond}(isubj,:) = nanmean(modcorrmat{icond});
    end
end

% ======= GENERATE PLOTE =======

% model and data on top of eachother (figure in manuscript
if (plot_datamodel_overlay)
    figure;
    for icond = 1:nConds
        
        % data
        m_data = mean(subjcorrMat{icond});
        sem_data = std(subjcorrMat{icond})./sqrt(nSubjs.(exptype));
        %     errorbar(m_data,sem_data,'Color',colorMat(icond,:)); hold on;
        
        % model
        m_mod= mean(modcorrMat{icond});
        sem_mod = std(modcorrMat{icond})./sqrt(nSubjs.(exptype));
        hold on;
        plot_summaryfit(1:nTimes,m_data,sem_data,m_mod,sem_mod,colorMat(icond,:),colorMat(icond,:));
    end
    ylim([0 1])
    xlim([0.5 11.5])
    set(gca,'XTick',0:11)
    title(model)
    xlabel('trials per stimuli')
    ylabel('proportion correct')
    defaultplot
end

% model and data next to eachother separately, one in each subplot
if (plot_datamodel_sep)
    figure;
    subplot(1,2,1)
    for icond = 1:nConds
        m_data = mean(subjcorrMat{icond});
        sem_data = std(subjcorrMat{icond})./sqrt(nSubjs.(exptype));
        errorbar(m_data,sem_data,'Color',colorMat(icond,:)); hold on;
    end
    ylim([0 1])
    xlim([0.5 11.5])
    set(gca,'XTick',0:11)
    title('data')
    xlabel('trials per stimuli')
    ylabel('proportion correct')
    defaultplot
    
    subplot(1,2,2)
    for icond = 1:nConds
        m_mod= mean(modcorrMat{icond});
        sem_mod = std(modcorrMat{icond})./sqrt(nSubjs.(exptype));
        errorbar(m_mod,sem_mod,'Color',colorMat(icond,:)); hold on;
    end
    ylim([0 1])
    xlim([0.5 11.5])
    set(gca,'XTick',0:11)
    title(model)
    xlabel('trials per stimuli')
    ylabel('proportion correct')
    defaultplot
end

% data for each participant
if (plot_indvlsubjects)
    figure;
    m = floor(sqrt(nSubjs.(exptype)));
    n = ceil(nSubjs.(exptype)/m);
    for isubj = 1:nSubjs.(exptype)
        subplot(m,n,isubj)
        for icond = 1:nConds
            plot(subjcorrMat{icond}(isubj,:),'Color',colorMat(icond,:)); hold on
        end
        defaultplot
    end
end

% model predictions for each participant
if (plot_indvlmodels)
    figure;
    for isubj = 1:nSubjs.(exptype)
        subplot(6,6,isubj)
        for icond = 1:nConds
            plot(modcorrMat{icond}(isubj,:),'Color',colorMat(icond,:)); hold on
        end
        defaultplot
    end
end

%% FIG 5(bottom): TEST PHASE MODEL VALIDATION

clear all

load('experimentalsettings.mat')
exptype = 'RPP';
nSubjs = nSubjs.(exptype);
subjidVec = subjidVec.(exptype);
nSims = 10;

% EXPERIMENT 2 MODELS (in addition to those above)
% model = 'RL3WM_all_pers_0';            % RL learning rate without interaction
% model = 'RL3WMi_all_pers_0';        % RL learning rate with interaction (fitted on learn+test data)
% model = 'RLWM_dn_all_pers_sub_0';
model = 'RLWMi_dn_all_pers_sub_0';  % RL decision noise with interaction (fitted on learn+test data)

% ----- load MLE parameters -----
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model))
[~,~,~,~,~,~,~,~,~,~,fixparams] = loadfittingparams(model);

% fix beta parameter to random value
fixparams(2,end) = 30;

% add in fixed parameter values
nps = size(xbestMat,2) + size(fixparams,2);
nv = 1:nps;
nv(fixparams(1,:)) = [];
xb(:,nv) = xbestMat;
xb(:,fixparams(1,:)) = repmat(fixparams(2,:),nSubjs,1);
xbestMat = xb;

% get model simulation model correct
if strcmp(model(end-1:end),'_0')
    modell = model(1:end-2);
    if strcmp(modell(end-3:end),'_sub')
        modell = modell(1:end-4);
    end
else
    modell = model;
end

try
    if strcmp(modell(end-8:end-5),'_all')
        modell = [modell(1:end-9) modell(end-4:end)];
    end
end

switch modell
    case {'RLWM_dn_pers','RLWMi_dn_pers'}
        modeltest = 'RLWMi_dn_pers';
    case {'RL3WM_pers','RL3WMi_pers'}
        modeltest = 'RL3WMi_pers';
    otherwise
        modeltest = modell;
        %     otherwise
        %         modeltest = 'nodn'; % any model not RLWM dn
end

for isubj = 1:nSubjs
    subjid = subjidVec(isubj);
    
    % ----- load data ------
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    % ----- get Q values for model simulated on real learning data -----
    [respCellCell, correctCell] = deal(cell(1,nSims));
    for isim = 1:nSims
        eval(sprintf('[~, QvalCell] = calc_LL_%s(xbestMat(isubj,:),stimvaluesCell,corrCell,responseCell,condVec);',modell))
        eval(sprintf('[~, correctCell{isim}] = simulatetest_%s(xbestMat(isubj,:),QvalCell,test_fullseq_learnblocknum,test_fullseq_stimvalues,test_fullseq_corrresp,condVec);',modeltest))
    end
    
    % ----- stuff for plotting -----
    [data_pc, model_pc] = deal(nan(2,nConds));
    for icond = 1:nConds
        idx_imagecond = condVec == conditionMat(2,icond);
        idx_ns = cellfun(@max,stimvaluesCell,'UniformOutput',true) == conditionMat(1,icond);
        blocknumVec = find(idx_imagecond & idx_ns);
        nBlocks = length(blocknumVec);
        
        for iblock = 1:nBlocks
            blocknum = blocknumVec(iblock);
            
            subj_respVec = test_subjrespCell{blocknum};
            idx = (subj_respVec ~= -1);
            data_pc(iblock,icond) = mean(test_corrrespCell{blocknum}(idx) == subj_respVec(idx));
            model_pc_temp = nan(1,nSims);
            for isim = 1:nSims
                model_pc_temp(isim) = mean(correctCell{isim}{blocknum}(idx));
            end
            model_pc(iblock,icond) = mean(model_pc_temp);
        end
    end
    data_pcMat(isubj,:) = mean(data_pc);
    model_pcMat(isubj,:) = mean(model_pc);
end

figure;
m_data = mean(data_pcMat);
sem_data = std(data_pcMat)./sqrt(nSubjs);
m_model = mean(model_pcMat);
sem_model = std(model_pcMat)./sqrt(nSubjs);
plotorder = [3 1 2];
hold on
for icond = 1:nConds
    errorbar(plotorder(conditionMat(2,icond)),m_data(icond),sem_data(icond),'Color',colorMat(icond,:),'CapSize',18);
    fill([-0.3 0.3 0.3 -0.3]+plotorder(conditionMat(2,icond)), m_model(icond)+[-1 -1 1 1].*sem_model(icond),colorMat(icond,:),'FaceAlpha',0.3,'EdgeColor','none');
end
defaultplot
set(gca,'XTick',1:3,'XTickLabel',{'Conjunctive','Linguistic','Visual'})
xlabel('condition')
ylabel('proportion correct')
ylim([1/3 1])
title(model)

%% FIG 2, 4, 16: MODEL COMPARISON


clear all
load('experimentalsettings.mat')

% ====== TOGGLE FOR DIFFERENT PLOTS ======

% Fig 2, 4: Experiment 1 and Experiment 2 replication
exptype = 'Mturk'; % or 'RPP' for Exp 2
modelVec = {'RL3WM_pers_0','RLWM_CA_pers_0','RLWM_dn_pers_sub_0','Decay3_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
n = 702; % number of trials

% % Experiment 2 (fit on learning+test data)
% exptype = 'RPP';
% modelVec = {'RL3WM_all_pers_0','RLWM_dn_all_pers_sub_0','RL3WMi_all_pers_0','RLWMi_dn_all_pers_sub_0'};
% n = 702+216; % number of trials

% % (Supplementary) Fig 16
% exptype = 'Mturk';
% modelVec = {'RL3WM_pers_0','RL_pers_0','WM_pers_0','RLWM_pers_0'};
% n = 702; % number of trials

% ======= CALCULATE \DELTA AICc AND BIC =====
imodelref = 1; % reference model (from which to subtract all other models)
nModels = length(modelVec); % number of models

nParamsVec = nan(1,nModels);
for imodel = 1:nModels
    model = modelVec{imodel};
    LB = loadfittingparams(model);
    nParamsVec(imodel) = length(LB);
end

LLMat = nan(nModels,nSubjs.(exptype));
for imodel = 1:nModels
    model = modelVec{imodel};
    
    load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'LLVec')
    LLMat(imodel,:) = LLVec;
end

AIC = 2*bsxfun(@plus,LLMat,nParamsVec');
BIC = bsxfun(@plus,2*LLMat,log(n)*nParamsVec');
AICc = bsxfun(@plus,AIC,2.*bsxfun(@rdivide,nParamsVec'.*(nParamsVec'+1),(bsxfun(@minus,n,nParamsVec')-1)));

Delta_AICc = bsxfun(@minus,AICc,AICc(imodelref,:));
Delta_BIC = bsxfun(@minus,BIC,BIC(imodelref,:));

med_AICc = median(Delta_AICc,2);
med_BIC = median(Delta_BIC,2);

% if all models same number of parameters, do LL comp
if (sum(nParamsVec == min(nParamsVec)) == length(nParamsVec))
    Delta_LL = bsxfun(@minus,LLMat,LLMat(imodelref,:));
    med_LL= median(Delta_LL,2);
    
    figure;
    for imodel = 1:nModels
        
        LLVec = Delta_LL(imodel,:);
        
        CI_LL = sort(median(LLVec(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
        CI_LL = CI_LL([25 975]);
        fill([0.55 1.45 1.45 0.55]+imodel-1,CI_LL([1 1 2 2]),0.7*ones(1,3)); hold on;
        plot([0.55 1.45]+imodel-1,[med_LL(imodel) med_LL(imodel)],'k-')
    end
    
    hold on;
    violinplot(Delta_LL');
    plot([0 nModels+0.5],[0 0],'k-')
    set(gca,'Xtick',1:nModels,'XTickLabel',modelVec)
    ylabel(sprintf('AICc(model) - AICc(%s)',modelVec{imodelref}))
    title('LL')
    defaultplot
    
else
    AIC = 2*bsxfun(@plus,LLMat,nParamsVec');
    BIC = bsxfun(@plus,2*LLMat,log(n)*nParamsVec');
    AICc = bsxfun(@plus,AIC,2.*bsxfun(@rdivide,nParamsVec'.*(nParamsVec'+1),(bsxfun(@minus,n,nParamsVec')-1)));
    
    Delta_AICc = bsxfun(@minus,AICc,AICc(imodelref,:));
    Delta_BIC = bsxfun(@minus,BIC,BIC(imodelref,:));
    
    med_AICc = median(Delta_AICc,2);
    med_BIC = median(Delta_BIC,2);
    
    figure;
    for imodel = 1:nModels
        
        daicc = Delta_AICc(imodel,:);
        dbic = Delta_BIC(imodel,:);
        
        subplot(1,2,1)
        CI_AICc = sort(median(daicc(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
        CI_AICc = CI_AICc([25 975]);
        fill([0.55 1.45 1.45 0.55]+imodel-1,CI_AICc([1 1 2 2]),0.7*ones(1,3)); hold on;
        plot([0.55 1.45]+imodel-1,[med_AICc(imodel) med_AICc(imodel)],'k-')
        
        subplot(1,2,2)
        CI_BIC = sort(median(dbic(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
        CI_BIC = CI_BIC([25 975]);
        fill([0.55 1.45 1.45 0.55]+imodel-1,CI_BIC([1 1 2 2]),0.7*ones(1,3)); hold on;
        plot([0.55 1.45]+imodel-1,[med_BIC(imodel) med_BIC(imodel)],'k-')
    end
    
    subplot(1,2,1); hold on;
    violinplot(Delta_AICc');
    plot([0 nModels+0.5],[0 0],'k-')
    set(gca,'Xtick',1:nModels,'XTickLabel',modelVec)
    ylabel(sprintf('AICc(model) - AICc(%s)',modelVec{imodelref}))
    title('AICc')
    defaultplot
    
    subplot(1,2,2); hold on;
    violinplot(Delta_BIC');
    plot([0 nModels+0.5],[0 0],'k-')
    set(gca,'Xtick',1:nModels,'XTickLabel',modelVec)
    ylabel(sprintf('BIC(model) - BIC(%s)',modelVec{imodelref}))
    title('BIC')
    defaultplot

end


%% =================================================================
%                  SUPPLEMENTARY FIGURES
% ==================================================================
%% FIG 7-12: PARAMETER RECOVERY PLOT

clear all

load('experimentalsettings.mat')
nsimSubjs = 50;

% fitting subset of total models
modelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
nModels = length(modelVec);
% nParamsVec = nParamsVec(imodelvec);
% paramnamesVec = paramnamesVec(imodelvec);

for imodel = 1:nModels
    model = modelVec{imodel};
    
    [~,~,~,~,logflag] = loadfittingparams(model);
    
    % load real parameteres
    for isubj = 1:nsimSubjs
        load(sprintf('data/simdata/simdata_model_%s_subj%d.mat',model,isubj))
        realparams(isubj,:) = x;
    end
    
    % load fitted parameters
    load(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s.mat',model,model))
    
    figure('Position',[550 250 840 320]);
    nParams = size(realparams,2);
    for iparam = 1:nParams
        subplot(2,4,iparam)
        
        minn = min([realparams(:,iparam); xbestMat(:,iparam)]);
        maxx = max([realparams(:,iparam); xbestMat(:,iparam)]);
        
        if logflag(iparam)
            plot(log(realparams(:,iparam)),log(xbestMat(:,iparam)),'ko'); hold on
            plot(log([minn maxx]),log([minn maxx]),'Color',0.7*ones(1,3));
        else
            plot(realparams(:,iparam),xbestMat(:,iparam),'ko'); hold on
            plot([minn maxx],[minn maxx],'Color',0.7*ones(1,3));
        end
        axis square
        axis equal
        axis square
        defaultplot
        xlabel('true')
        ylabel('estimated')
        %         title(paramnamesVec{imodel}{iparam})
        if (iparam == 1)
            title(model)
        end
    end
    
    clear realparams
    %     biglabelplot(model)
    
end

%% FIG 13: MODEL RECOVERY PLOT

clear all

load('experimentalsettings.mat')
nsimSubjs = 50;

% subset of total models
modelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
nModels = length(modelVec);
nParamsVec = nan(1,nModels);
% nParamsVec = nParamsVec(imodelvec);

[modelrecovMat_AICc, modelrecovMat_BIC] = deal(nan(nModels,nModels));
for isimmodel = 1:nModels
    simmodel = modelVec{isimmodel};
    
    LLMat = nan(nModels,nsimSubjs);
    for ifitmodel = 1:nModels
        fitmodel = modelVec{ifitmodel};
        
        load(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s.mat',fitmodel,simmodel));
        LLMat(ifitmodel,:) = LLVec;
        if (isimmodel == 1)
            nParamsVec(ifitmodel) = size(xbestMat,2);
        end
    end
    
    AIC = 2*bsxfun(@plus,LLMat,nParamsVec');
    BIC = bsxfun(@plus,2*LLMat,log(nTrials)*nParamsVec');
    AICc = bsxfun(@plus,AIC,2.*bsxfun(@rdivide,nParamsVec'.*(nParamsVec'+1),(bsxfun(@minus,nTrials,nParamsVec')-1)));
    
    modelrecovMat_AICc(:,isimmodel) = sum(bsxfun(@(x,y) x==y,min(AICc),AICc),2);
    modelrecovMat_BIC(:,isimmodel) = sum(bsxfun(@(x,y) x==y,min(BIC),BIC),2);
    
end

% generating some variables for labelling each cell
[x,y] = meshgrid(1:nModels,1:nModels);

figure;
imagesc(modelrecovMat_AICc,[0 nsimSubjs]);
tt = num2cell(modelrecovMat_AICc);
tt = cellfun(@num2str, tt, 'UniformOutput', false);
text(x(:), y(:), tt, 'HorizontalAlignment', 'Center')
colormap('parula')
set(gca,'XTick',1:nModels,'XTickLabel',modelVec,'YTick',1:nModels,'YTickLabel',modelVec)
ylabel('winning model')
xlabel('true model')
title('AICc')
colorbar
defaultplot

figure;
imagesc(modelrecovMat_BIC,[0 nsimSubjs]);
tt = num2cell(modelrecovMat_BIC);
tt = cellfun(@num2str, tt, 'UniformOutput', false);
text(x(:), y(:), tt, 'HorizontalAlignment', 'Center')
colormap('parula')
set(gca,'XTick',1:nModels,'XTickLabel',modelVec,'YTick',1:nModels,'YTickLabel',modelVec)
ylabel('winning model')
xlabel('true model')
title('BIC')
colorbar
defaultplot

%% FIG 14-15: COMPARE PARAMETERS ACROSS PARTICIPANTS AND EXPERIMENTS

clear all

load('experimentalsettings.mat')


% model = 'RLWM_dn_pers_sub_0';
model = 'RL3WM_pers_0';
paramnames = paramnamesVec.(model);
[~,~,~,~,logflag] = loadfittingparams(model);

% get parameters
exptype = 'RPP';
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat','LLVec')
xb_rpp = xbestMat;
xb_rpp(:,logflag) = log(xb_rpp(:,logflag));
m_rpp = mean(xb_rpp);
sem_rpp = std(xb_rpp)./sqrt(nSubjs.(exptype));

exptype = 'Mturk';
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat','LLVec')
xb_mturk = xbestMat;
xb_mturk(:,logflag) = log(xb_mturk(:,logflag));
m_mturk = mean(xb_mturk);
sem_mturk = std(xb_mturk)./sqrt(nSubjs.(exptype));

% plot

nparams = length(m_rpp);
m = floor(sqrt(nparams));
n = ceil(nparams/m);

figure
for iparam = 1:nparams
    
    % ttest
    [h(iparam),p(iparam),ci{iparam},stats{iparam}] = ttest2(xb_rpp(:,iparam),xb_mturk(:,iparam));

    % plot
    subplot(m,n,iparam); hold on;
    blah.b1 = xb_mturk(:,iparam);
    blah.b2 = xb_rpp(:,iparam);
    violinplot(blah);
    errorbar([1 2],[m_mturk(iparam) m_rpp(iparam) ],...
        [sem_mturk(iparam) sem_rpp(iparam)],'k','LineStyle','none','CapSize',14)
    set(gca,'Xtick',[])
    if logflag(iparam)
        title(sprintf('log(%s)',paramnames{iparam}))
    else
        title(paramnames(iparam))
    end
    text(0.8,max(get(gca,'Ylim')),sprintf('p=%0.3f',p(iparam)))
    defaultplot
    if iparam > n
        xlabel('experiment')
        set(gca,'XTick',[1 2],'XTickLabel',{'1: Mturk','2: RPP'})
    end
    if (mod(iparam,n) == 1); ylabel('value'); end
    
end


%% FIG 17: FACTORIAL MODEL COMPARISON
% of main six models with/without perseveration and/or alpha-

clear all
load('experimentalsettings.mat')
exptype = 'Mturk';

submodelVec = {'RL3WM','RLWM_CA','RLWM_dn_sub','Decay3','RLWM_dnwm_sub','ns2'};
nModels = length(submodelVec);
imodelref = 19;


modelnameVec = [];
for iflag_pers = 1:2
    flag_pers = logical(iflag_pers-1);
    
    for iflag_alphaneg = 1:2
        flag_alphaneg = logical(iflag_alphaneg-1);
        
        for imodel = 1:nModels
            model = submodelVec{imodel};
            
            if strcmp(model(end-2:end),'sub') % if not fitting a dn/CA parameter for category condition
                flag_dnsub = 1;
            else
                flag_dnsub = 0;
            end
            
            if (flag_pers) % if adding perseveration
                if (flag_dnsub)
                    model = [model(1:end-4) '_pers' '_sub'];
                else
                    model = [model '_pers'];
                end
            end
            
            if (flag_alphaneg) % if adding alpha- = 0
                model = [model '_0'];
            end
            
            modelnameVec = [modelnameVec {model}];
        end
    end
end

nModels = length(modelnameVec);

LLMat = nan(nModels,nSubjs.(exptype));
nParamsVec = nan(1,nModels);
for imodel = 1:nModels
    model = modelnameVec{imodel};
    
    load(sprintf('fits/%s/fits_model_%s.mat',exptype,model))
    LLMat(imodel,:) = LLVec;
    nParamsVec(imodel) = size(xbestMat,2);
end

AIC = 2*bsxfun(@plus,LLMat,nParamsVec');
BIC = bsxfun(@plus,2*LLMat,log(nTrials)*nParamsVec');
AICc = bsxfun(@plus,AIC,2.*bsxfun(@rdivide,nParamsVec'.*(nParamsVec'+1),(bsxfun(@minus,nTrials,nParamsVec')-1)));

Delta_AICc = bsxfun(@minus,AICc,AICc(imodelref,:));
Delta_BIC = bsxfun(@minus,BIC,BIC(imodelref,:));

med_AICc = median(Delta_AICc,2);
med_BIC = median(Delta_BIC,2);

for imodel = 1:nModels
    
    daicc = Delta_AICc(imodel,:);
    dbic = Delta_BIC(imodel,:);
    
    figure(1);
    CI_AICc = sort(median(daicc(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
    CI_AICc = CI_AICc([25 975]);
    fill([0.55 1.45 1.45 0.55]+imodel-1,CI_AICc([1 1 2 2]),0.7*ones(1,3)); hold on;
    plot([0.55 1.45]+imodel-1,[med_AICc(imodel) med_AICc(imodel)],'k-')
    
    figure(2);
    CI_BIC = sort(median(dbic(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
    CI_BIC = CI_BIC([25 975]);
    fill([0.55 1.45 1.45 0.55]+imodel-1,CI_BIC([1 1 2 2]),0.7*ones(1,3)); hold on;
    plot([0.55 1.45]+imodel-1,[med_BIC(imodel) med_BIC(imodel)],'k-')
end

figure(1);
violinplot(Delta_AICc');
plot([0 nModels+0.5],[0 0],'k-')
set(gca,'Xtick',1:nModels,'XTickLabel',modelnameVec)
ylabel(sprintf('AICc(model) - AICc(%s)',modelnameVec{imodelref}))
title('AICc')
defaultplot

figure(2);
violinplot(Delta_BIC');
plot([0 nModels+0.5],[0 0],'k-')
set(gca,'Xtick',1:nModels,'XTickLabel',modelnameVec)
ylabel(sprintf('BIC(model) - BIC(%s)',modelnameVec{imodelref}))
title('BIC')
defaultplot

%% FIG 18: FACTORIAL MODEL COMPARISON
% of main six models with/without full perseveration

clear all
load('experimentalsettings.mat')
exptype = 'Mturk';

flag_alphaneg = 1;
submodelVec = {'RL3WM','RLWM_CA','RLWM_dn_sub','Decay3','RLWM_dnwm_sub','ns2'};
nModels = length(submodelVec);
imodelref = 7;

modelnameVec = [];


for iflag_pers = 1:2
    flag_pers = logical(iflag_pers-1);
    for imodel = 1:nModels
        model = submodelVec{imodel};
        
        
        
        if (flag_pers) % if adding full perseveration
            persstr = '_fullpers';
        else
            persstr = '_pers';
        end
        
        if strcmp(model(end-2:end),'sub') % if not fitting a dn/CA parameter for category condition
            model = [model(1:end-4) persstr '_sub'];
        else
            model = [model persstr];
        end
        
        if (flag_alphaneg) % if adding alpha- = 0
            model = [model '_0'];
        end
        
        modelnameVec = [modelnameVec {model}];
    end
end

nModels = length(modelnameVec);

LLMat = nan(nModels,nSubjs.(exptype));
nParamsVec = nan(1,nModels);
for imodel = 1:nModels
    model = modelnameVec{imodel};
    
    load(sprintf('fits/%s/fits_model_%s.mat',exptype,model))
    LLMat(imodel,:) = LLVec;
    nParamsVec(imodel) = size(xbestMat,2);
end

AIC = 2*bsxfun(@plus,LLMat,nParamsVec');
BIC = bsxfun(@plus,2*LLMat,log(nTrials)*nParamsVec');
AICc = bsxfun(@plus,AIC,2.*bsxfun(@rdivide,nParamsVec'.*(nParamsVec'+1),(bsxfun(@minus,nTrials,nParamsVec')-1)));

Delta_AICc = bsxfun(@minus,AICc,AICc(imodelref,:));
Delta_BIC = bsxfun(@minus,BIC,BIC(imodelref,:));

med_AICc = median(Delta_AICc,2);
med_BIC = median(Delta_BIC,2);

figure;
for imodel = 1:nModels
    
    daicc = Delta_AICc(imodel,:);
    dbic = Delta_BIC(imodel,:);
    
    subplot(1,2,1)
    CI_AICc = sort(median(daicc(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
    CI_AICc = CI_AICc([25 975]);
    fill([0.55 1.45 1.45 0.55]+imodel-1,CI_AICc([1 1 2 2]),0.7*ones(1,3)); hold on;
    plot([0.55 1.45]+imodel-1,[med_AICc(imodel) med_AICc(imodel)],'k-')
    
    subplot(1,2,2)
    CI_BIC = sort(median(dbic(randi(nSubjs.(exptype),nSubjs.(exptype),1000))));
    CI_BIC = CI_BIC([25 975]);
    fill([0.55 1.45 1.45 0.55]+imodel-1,CI_BIC([1 1 2 2]),0.7*ones(1,3)); hold on;
    plot([0.55 1.45]+imodel-1,[med_BIC(imodel) med_BIC(imodel)],'k-')
end

subplot(1,2,1); hold on;
violinplot(Delta_AICc');
plot([0 nModels+0.5],[0 0],'k-')
set(gca,'Xtick',1:nModels,'XTickLabel',modelnameVec)
ylabel(sprintf('AICc(model) - AICc(%s)',modelnameVec{imodelref}))
title('AICc')
defaultplot

subplot(1,2,2); hold on;
violinplot(Delta_BIC');
plot([0 nModels+0.5],[0 0],'k-')
set(gca,'Xtick',1:nModels,'XTickLabel',modelnameVec)
ylabel(sprintf('BIC(model) - BIC(%s)',modelnameVec{imodelref}))
title('BIC')
defaultplot


%% COMPARE PARAMETERS OF SIMULATED VS REAL DATA

clear all
close all

nsimSubjs = 50;
load('experimentalsettings.mat')
exptypeCell = {'RPP','Mturk'};

modelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
nModels = length(modelVec);
for imodel = 1:nModels
    model = modelVec{imodel}
    
    xb = [];
    for iexp = 1:2
        exptype = exptypeCell{iexp};
        load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat')
        xb = [xb; xbestMat];
    end
    xbestMat = xb;
    nParams = size(xb,2);
    nSubjs = size(xb,1);
    
    xx = nan(nsimSubjs,nParams);
    for isubj = 1:nsimSubjs
        load(sprintf('data/simdata/simdata_model_%s_subj%d.mat',model,isubj),...
            'x');
        xx(isubj,:) = x;
    end
    
    n = ceil(sqrt(nParams));
    
    figure;
    title(model)
    for iparam = 1:nParams
        subplot(n,n,iparam)
        plot(zeros(1,nSubjs),xbestMat(:,iparam),'ko')
        hold on;
        plot(ones(1,nsimSubjs),xx(:,iparam),'ko')
        %             title(paramnamesVec{imodel}{iparam})
        xlim([-0.5 1.5])
        defaultplot
        set(gca,'XTick',[0,1],'XTickLabel',{'real','simulated'})
        
    end
    %         biglabelplot(model)
end

