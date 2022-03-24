function [respCell, correctCell] = simulate_Decay3(pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_DECAY3 simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x9 vector of parameters 
%    [alpha, epsilon, lambda_e, lambda_c, lambda_t, ns3, ns6, beta]
%       alpha: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
%       lambda_e: WM decay rate for exemplar condition blocks
%       lambda_c: WM decay rate for category condition blocks
%       lambda_t: WM decay rate for text condition blocks
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       beta: softmax inverse temperature parameter
% STIMVALUESCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars 
%       corresponding to the index of the stimulus presented on each trial
% CORRRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of lanth nTrials of scalars
%       corresponding to the index of the correct button response
% CONDVEC: 1 x nBlocks vector, condition indices 
% SUBJRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the participant's response
%
% ========================= OUTPUT VARIABLES ============================
% RESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the index of the simulated button response
% CORRECTCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to whether the simulated data was rewarded for
%       correct response. 
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu

% pars
alpha = pars(1);
neg_alpha = pars(2);
epsilon = pars(3);
lambda = pars(4:6);
ns = pars(7:8);
beta_RL = pars(9);
beta_WM = 100;
nBlocks = length(stimvaluesCell);

[respCell, correctCell] = deal(cell(1,nBlocks));
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    icond = condVec(iblock);
    
    % data
    corrrespVec = corrrespCell{iblock};
    
    [respVec,correctVec] = deal(nan(1,nTrials));
    [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses    
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        
        % p of each choice
        p_RL = exp(beta_RL*QvalMat(istim,:)); p_RL=p_RL/sum(p_RL);
        p_WM = exp(beta_WM*WMvalMat(istim,:)); p_WM=p_WM/sum(p_WM);
        p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL;
        p = (1-epsilon).*p + epsilon*1/3;
      
        if find(stimVec == istim,1,'first') == itrial;
            ichoice = subjrespCell{iblock}(itrial);
        else
            ichoice = find([0 cumsum(p)]<rand,1,'last');
        end
        r = (ichoice == corrrespVec(itrial));
        
        if (ichoice ~= -1)
            % updating
            if (r==1);
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*(r-QvalMat(istim,ichoice));
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*(r-QvalMat(istim,ichoice));
            end
            WMvalMat = (1-lambda(icond)).*1/3 + lambda(icond).*WMvalMat; % decay
            WMvalMat(istim,ichoice) = r;
        end
        
        % saving variables
        respVec(itrial) = ichoice;
        correctVec(itrial) = r;
    end
    
    respCell{iblock} = respVec;
    correctCell{iblock} = correctVec;
end