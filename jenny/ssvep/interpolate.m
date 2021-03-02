
function interpolate(subID)
    % input:  e.g., 's181_ses1'
    % perform a spline interpolation on the pdmattention data
    % good channels are identified when it's identifed non-artfifact for 
    % for less than 40 trials
    fprintf('Current Subject: %s\n\n', subID);
    basedir = '/home/jenny/pdmattention/';
    datadir = append(basedir,'task3');
    specdir = append(basedir,'task3/spectrogram');
    cd(datadir);
    datain = load(append(subID, '_task3_final.mat'));
    behavdata = load(append(subID(1:4), '_behavior_final1.mat'));
    sr = datain.sr;
    rt = behavdata.rt;
    correct = behavdata.correct;
    condition = behavdata.condition;
    
    final_interp = struct();

    chanpos = datain.hm.Electrode.CoordOnSphere;
    if isfield(datain.hm.Electrode,'NoInterp')
        interpchans=setdiff(1:size(chanpos,1),datain.hm.Electrode.NoInterp);
    else
        interpchans = 1:size(chanpos,1);
    end

    % identify bad channels using the < 40 criteria
    artifact = datain.artifact;
    artifact1 = sum(artifact,1); % a 360 row vector    
    artifact2 = sum(artifact,2); % a 128 row vector 

    trials = behavdata.trials +1;
    goodtrials = find(artifact1<20);
    if all(ismember(trials, goodtrials))
        goodtrials = goodtrials;
        final_interp.trialflag = 0;
    else
        goodtrials = trials(ismember(trials, goodtrials));
        fprintf('warning: trials re-selected');
        final_interp.trialflag = 1
    end
    data= nan(4000,129,360);

    for t = goodtrials
        mat=splineinterp(.1,chanpos(interpchans(datain.artifact(interpchans,t)==0),:),chanpos(interpchans,:));
        data(:,interpchans,t)=datain.data(:,interpchans(datain.artifact(interpchans,t)==0),t)*mat';
        final_interp.interp(interpchans(datain.artifact(interpchans,t)==1),t) = 1;
    end
    final_interp.data = data;
    final_interp.artifact = artifact;
    final_interp.trials = goodtrials;
    final_interp.sr = sr;
    final_interp.rt = rt;
    final_interp.correct = correct;
    final_interp.condition = condition;
    cd('/home/jenny/pdmattention/task3/final_interp');
    fname = sprintf('%s_final_interp.mat', subID);
    save(fname, '-struct','final_interp');
end