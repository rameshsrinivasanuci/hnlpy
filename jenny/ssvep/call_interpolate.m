% loop through all the subjects for pdmattention
path = '/home/jenny/pdmattention/task3';
cd(path);
MyFileInfo = dir('*expinfo.mat*');

subIDs = vertcat(MyFileInfo.name);
subIDs = subIDs(:,1:9);             % get subIDS as sxxx_sesx'

for i = 1:length(subIDs)
    sub = subIDs(i,:);
    interpolate(sub)
    fprintf('%s saved\n', sub)
end
