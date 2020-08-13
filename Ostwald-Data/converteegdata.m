function converteegdata(subID)
% subID as 'sub-xxx'
% this function loads the Ostwald EEG/fMRI data using eeglab2020
% this is the clean eeg data inside the MRI under the folder
% input path '../derivatives/cleandata-eeg_inside-MRT'
% output paht '../derivatives/cleandata-eeg_inside-MRT/clean-eeg-converted'

fprintf('%s\n\n', subID)
basedir = '/home/jenny/ostwald-data/derivatives/cleandata-eeg_inside-MRT';
subjectdir = fullfile(basedir, subID, 'eeg');
cd(subjectdir)
MyFileInfo = dir('*.vhdr*');
FileName = vertcat(MyFileInfo.name);
NumFile = size(FileName, 1);

if NumFile < 1
   fprintf('Warning: no run detected for subject %s\n', subID);
else
    fprintf('%d runs detected for %s\n', NumFile, subID)
end

for i = 1:NumFile
    outputdata = pop_loadbv(subjectdir, FileName(i,:));   
    outputdir = fullfile(basedir,'clean-eeg-converted');
    NewFileName = fullfile(outputdir, strcat(FileName(i,1:7),FileName(i,end-15:end-5),'.mat'));
    jsonfname = fullfile(subjectdir, strcat(FileName(i, 1:end-5),'.json'));
    if sum(jsonfname ~= strcat(subjectdir,'/sub-001_task-pdm_acq-insideMRT_run-02_eeg.json')) > 0
       jsonfile = jsondecode(fileread(jsonfname));
       save(NewFileName,'outputdata','jsonfile')
    else
       fprintf('this is an error file')
       save(NewFileName,'outputdata')
    end
    fprintf('\n\n%s saved\n\n\n',NewFileName(end-21:end))
end