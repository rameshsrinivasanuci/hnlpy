function convertall
% This function runs through each subject under '../derivatives/cleandata-eeg_inside-MRT'

basedir = '/home/jenny/ostwald-data/derivatives/cleandata-eeg_inside-MRT';
subjectList =  dir(basedir);
subjectList = {subjectList.name};
ind = strfind(dirs,'sub');
ind = find(not(cellfun('isempty',ind)));
subjectList = vertcat(subjectList(ind));

for i = 1: size(subjectList,2)
    converteegdata(cell2mat(subjectList(i)))
end


   