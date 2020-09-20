%mms2020_PlotPCA_Ostwald.m Plots topos of components from Bayesian PCA (see pystan_Bayesian_PCA_n200_vb.py)
%
%% Record of revisions:
%   Date           Programmers               Description of change
%   ====        =================            =====================
%  08/25/20     Michael Nunez             Converted from mms2020_plotPCA_Ostwald
%%%%%%%%%%%%%%%%%%%%%%
%% Initial
addpath(genpath('/home/michael/artscreenEEG'));
load extnd1020hm.mat;
samples = load('/home/ramesh/pdmattention/scripts/model_fits/Bayesian_PCA_n200Aug_26_20_23_slide8.mat');
chanindx = load('/home/ramesh/pdmattention/scripts/model_fits/sub006_newchan.mat');
previousgoodchan = chanindx.goodchan+1;
truelocations = load('/home/ramesh/pdmattention/scripts/model_fits/pdm_erp_electrode_locations.mat');
BRAINAMP64 = EXTND1020;
for n=1:length(truelocations.chanlocs),
BRAINAMP64.ChanNames{n} = '?';
BRAINAMP64.Electrode.CoordOnSphere(n,1:3) = zeros(1,3);
end
for n=1:length(truelocations.chanlocs),
for m=1:length(EXTND1020.ChanNames),
if strcmpi(truelocations.chanlocs(n).labels,EXTND1020.ChanNames{m}),
BRAINAMP64.ChanNames{n} = EXTND1020.ChanNames{m};
BRAINAMP64.Electrode.CoordOnSphere(n,:) = EXTND1020.Electrode.CoordOnSphere(m,:);
end
end
end

badchan = [];
for n=1:length(BRAINAMP64.ChanNames),
if strcmpi(BRAINAMP64.ChanNames{n},'?')
badchan = [badchan n];
end
end

goodchan = setdiff(previousgoodchan,badchan);

% for k=1:length(BRAINAMP64.ChanNames),
% 	if strcmpi(truelocations.chanlocs(n).labels,'EOG'),
% 		BRAINAMP64.ChanNames{n} = [];


%%%%%%%%%%%%%%%%%%%%%%
%% Code

f1 = figure('units','normalized','outerposition',[0 0 2 2]);
subplot(1,2,1);
tempweights = zeros(1, 64);
tempweights(previousgoodchan) = samples.comp1_weights;
% tempweights(previousgoodchan) = normrnd(0,1,1,length(previousgoodchan));
hortontopo(tempweights,BRAINAMP64,'goodchans',goodchan, 'drawelectrodes', 0, 'channumbers', 0, ...
    'cmap','parula');
subplot(1,2,2);
tempweights = zeros(1, 64);
tempweights(previousgoodchan) = samples.comp2_weights;
% tempweights(previousgoodchan) = normrnd(0,1,1,length(previousgoodchan));
hortontopo(tempweights,BRAINAMP64,'goodchans',goodchan, 'drawelectrodes', 0, 'channumbers', 0, ...
    'cmap','parula');
% subplot(1,3,3);
% tempweights = zeros(1, 64);
% tempweights(previousgoodchan) = samples.comp3_weights;
% % tempweights(previousgoodchan) = normrnd(0,1,1,length(previousgoodchan));
% hortontopo(tempweights,BRAINAMP64,'goodchans',goodchan, 'drawelectrodes', 0, 'channumbers', 0, ...
%     'cmap','parula');


addpath(genpath('/home/michael/MATLAB/altmany-export_fig-ee6506f'));
export_fig(f1,['sub006_Bayesian_PCA_N200_Ostwald'],'-opengl','-eps','-r300');
export_fig(f1,['sub006_Bayesian_PCA_N200_Ostwald'],'-opengl','-png','-r300');