function main_signature_verification_compcorr_cross_dataset(dataset1,dataset2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% FIRST SET THE PATHS %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dir_libsvm = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/libsvm/matlab'; % path to matlab folder of libsvm
% dir_liblinear = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/liblinear/matlab'; % path to matlab folder of liblinear
% dir_vlfeat = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/vlfeat'; % path to vlfeat

dir_libsvm = '/home/anjan/Dropbox/Personal/Workspace/AdditionalTools/libsvm/matlab'; % path to matlab folder of libsvm
dir_liblinear = '/home/anjan/Dropbox/Personal/Workspace/AdditionalTools/liblinear/matlab'; % path to matlab folder of liblinear
dir_vlfeat = '/home/anjan/Dropbox/Personal/Workspace/AdditionalTools/vlfeat'; % path to vlfeat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nclasses = 2; % number of classes of classification problem
Cs = 2.^(-7:2:9); % SVM C parameter
opt_str_liblinear = '-s 0 -v 5 -c %f -e 0.0001 -q'; % LIBLINEAR option string
rng(0); % seeding randomization
niter = 10;
% dataset = 'Hindi';
parts = strsplit(pwd, '/');
Signsroot = fullfile('/',parts{1:end-1}); % parent folder

%%%%%%%%%%%%%%%%%%%%%%%%%%% Force computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

force_compute.hist_indices = true;
force_compute.hists = true;
force_compute.kernels = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% add path 
addpath(dir_libsvm);
addpath(dir_liblinear);
addpath(genpath(dir_vlfeat));

% remove some of the added vlfeat paths to avoid conflicts
rmpath([dir_vlfeat,'/toolbox/kmeans']);
rmpath([dir_vlfeat,'/toolbox/gmm']);
rmpath([dir_vlfeat,'/toolbox/fisher']);
rmpath([dir_vlfeat,'/toolbox/noprefix']);

% file & directory names

if(strcmp(dataset1,dataset2))
    
    switch dataset1

        case 'CEDAR' % CEDAR dataset        

            subdir1 = 'Datasets/CEDAR/full_org';
            subdir2 = 'Datasets/CEDAR/full_forg';

            % source files
            images1 = dir(fullfile(Signsroot,subdir1,'*.png'));
            writers1 = single(cellfun(@str2num,strtok(strrep(strrep({images1.name},'original_',''),'.png',''),'_')));
            images1 = cellfun(@(x)fullfile(Signsroot,subdir1,x),{images1.name},'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = dir(fullfile(Signsroot,subdir2,'*.png'));
            writers2 = -single(cellfun(@str2num,strtok(strrep(strrep({images2.name},'forgeries_',''),'.png',''),'_')));
            images2 = cellfun(@(x)fullfile(Signsroot,subdir2,x),{images2.name},'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1,images2]'; clear images1 images2;
            all_writers1 = [writers1,writers2]'; clear writers1 writers2;

            train_test_ratio1 = 0.9; % train test ratio
            percent_dataset1 = 0.3; % percentage of training and test data
            
            tot_sigs_pair21 = tot_sigs_pair11;
            tot_sigs_pair22 = tot_sigs_pair12;
            all_images2 = all_images1;
            all_writers2 = all_writers1;
            train_test_ratio2 = train_test_ratio1;
            percent_dataset2 = percent_dataset1;

        case 'GPDS300' % GPDS300 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.5; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data
            
            tot_sigs_pair21 = tot_sigs_pair11;
            tot_sigs_pair22 = tot_sigs_pair12;
            all_images2 = all_images1;
            all_writers2 = all_writers1;
            train_test_ratio2 = train_test_ratio1;
            percent_dataset2 = percent_dataset1;

        case 'GPDS960' % GPDS960 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.5; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data
            
            tot_sigs_pair21 = tot_sigs_pair11;
            tot_sigs_pair22 = tot_sigs_pair12;
            all_images2 = all_images1;
            all_writers2 = all_writers1;
            train_test_ratio2 = train_test_ratio1;
            percent_dataset2 = percent_dataset1;

        case 'Bengali'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.8; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data
            
            tot_sigs_pair21 = tot_sigs_pair11;
            tot_sigs_pair22 = tot_sigs_pair12;
            all_images2 = all_images1;
            all_writers2 = all_writers1;
            train_test_ratio2 = train_test_ratio1;
            percent_dataset2 = percent_dataset1;

        case 'Hindi'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.8; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data
            
            tot_sigs_pair21 = tot_sigs_pair11;
            tot_sigs_pair22 = tot_sigs_pair12;
            all_images2 = all_images1;
            all_writers2 = all_writers1;
            train_test_ratio2 = train_test_ratio1;
            percent_dataset2 = percent_dataset1;

        otherwise        
            error('Wrong dataset');

    end;
    
    file1.hist_indices = fullfile(Signsroot,'SavedData',dataset1,'hist_indices_compcorr.mat');
    file1.hists = fullfile(Signsroot,'SavedData',dataset1,'hists_compcorr.mat');
    
    file2.hist_indices = fullfile(Signsroot,'SavedData',dataset2,'hist_indices_compcorr.mat');
    file2.hists = fullfile(Signsroot,'SavedData',dataset2,'hists_compcorr.mat');

    nimages1 = size(all_images1,2);
    org_writers1 = unique(abs(all_writers1));
    norg_writers1 = length(org_writers1);
    
    nimages2 = size(all_images2,2);
    org_writers2 = unique(abs(all_writers2));
    norg_writers2 = length(org_writers2);
    
%%%%%% Prepare niter sets of training and test writers from dataset1 %%%%%%
    
    train_writers = cell(1,niter);
    test_writers = cell(1,niter);
    for i = 1:niter
        train_writers{i} = single(sort(randsample(1:norg_writers1,round(train_test_ratio1*norg_writers1))));    
        test_writers{i} = single(setdiff(1:norg_writers1,train_writers{i}));    
    end;
    ntrain_writers = length(train_writers{1});
    ntest_writers = length(test_writers{1});
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
else

% dataset 1

    switch dataset1

        case 'CEDAR' % CEDAR dataset        

            subdir1 = 'Datasets/CEDAR/full_org';
            subdir2 = 'Datasets/CEDAR/full_forg';

            % source files
            images1 = dir(fullfile(Signsroot,subdir1,'*.png'));
            writers1 = single(cellfun(@str2num,strtok(strrep(strrep({images1.name},'original_',''),'.png',''),'_')));
            images1 = cellfun(@(x)fullfile(Signsroot,subdir1,x),{images1.name},'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = dir(fullfile(Signsroot,subdir2,'*.png'));
            writers2 = -single(cellfun(@str2num,strtok(strrep(strrep({images2.name},'forgeries_',''),'.png',''),'_')));
            images2 = cellfun(@(x)fullfile(Signsroot,subdir2,x),{images2.name},'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1,images2]'; clear images1 images2;
            all_writers1 = [writers1,writers2]'; clear writers1 writers2;

            train_test_ratio1 = 0.9; % train test ratio
            percent_dataset1 = 0.3; % percentage of training and test data           

        case 'GPDS300' % GPDS300 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.5; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data
            
        case 'GPDS960' % GPDS960 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.5; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data

        case 'Bengali'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.8; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data

        case 'Hindi'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair11 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair12 = sigs_per_writer1*sigs_per_writer2;

            all_images1 = [images1;images2]; clear images1 images2;
            all_writers1 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio1 = 0.8; % train test ratio
            percent_dataset1 = 0.02; % percentage of training and test data

        otherwise        
            error('Wrong dataset');

    end;

    file1.hist_indices = fullfile(Signsroot,'SavedData',dataset1,'hist_indices_compcorr.mat');
    file1.hists = fullfile(Signsroot,'SavedData',dataset1,'hists_compcorr.mat');

    nimages1 = size(all_images1,2);
    org_writers1 = unique(abs(all_writers1));
    norg_writers1 = length(org_writers1);

    %%%%%%%%%%% Prepare niter sets of training writers from dataset1 %%%%%%%%%%

    train_writers = cell(1,niter);
    for i = 1:niter
        train_writers{i} = single(sort(randsample(1:norg_writers1,round(train_test_ratio1*norg_writers1))));    
    end;
    ntrain_writers = length(train_writers{1});

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % dataset 2

    switch dataset2

        case 'CEDAR' % CEDAR dataset        

            subdir1 = 'Datasets/CEDAR/full_org';
            subdir2 = 'Datasets/CEDAR/full_forg';

            % source files
            images1 = dir(fullfile(Signsroot,subdir1,'*.png'));
            writers1 = single(cellfun(@str2num,strtok(strrep(strrep({images1.name},'original_',''),'.png',''),'_')));
            images1 = cellfun(@(x)fullfile(Signsroot,subdir1,x),{images1.name},'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair21 = nchoosek(sigs_per_writer1,2);

            images2 = dir(fullfile(Signsroot,subdir2,'*.png'));
            writers2 = -single(cellfun(@str2num,strtok(strrep(strrep({images2.name},'forgeries_',''),'.png',''),'_')));
            images2 = cellfun(@(x)fullfile(Signsroot,subdir2,x),{images2.name},'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair22 = sigs_per_writer1*sigs_per_writer2;

            all_images2 = [images1,images2]'; clear images1 images2;
            all_writers2 = [writers1,writers2]'; clear writers1 writers2;

            train_test_ratio2 = 0.9; % train test ratio
            percent_dataset2 = 0.3; % percentage of training and test data

        case 'GPDS300' % GPDS300 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS300/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair21 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair22 = sigs_per_writer1*sigs_per_writer2;

            all_images2 = [images1;images2]; clear images1 images2;
            all_writers2 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio2 = 0.5; % train test ratio
            percent_dataset2 = 0.02; % percentage of training and test data

        case 'GPDS960' % GPDS960 dataset        

            fp1 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/GPDS960/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);
            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair21 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS960',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair22 = sigs_per_writer1*sigs_per_writer2;

            all_images2 = [images1;images2]; clear images1 images2;
            all_writers2 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio2 = 0.5; % train test ratio
            percent_dataset2 = 0.02; % percentage of training and test data

        case 'Bengali'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Bengali/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair21 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair22 = sigs_per_writer1*sigs_per_writer2;

            all_images2 = [images1;images2]; clear images1 images2;
            all_writers2 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio2 = 0.8; % train test ratio
            percent_dataset2 = 0.02; % percentage of training and test data

        case 'Hindi'

            fp1 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.genuine'));
            images1 = textscan(fp1,'%s');
            fclose(fp1);
            fp2 = fopen(fullfile(Signsroot,'/Datasets/Hindi/list.forgery'));
            images2 = textscan(fp2,'%s');
            fclose(fp2);

            images1 = images1{:};
            writers1 = single(cellfun(@str2num,strtok(images1,'/')));
            images1 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images1,'UniformOutput',false);
            sigs_per_writer1 = min(histc(writers1, unique(writers1)));
            tot_sigs_pair21 = nchoosek(sigs_per_writer1,2);

            images2 = images2{:};
            writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
            images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images2,'UniformOutput',false);
            sigs_per_writer2 = min(histc(writers2, unique(writers2)));
            tot_sigs_pair22 = sigs_per_writer1*sigs_per_writer2;

            all_images2 = [images1;images2]; clear images1 images2;
            all_writers2 = [writers1;writers2]; clear writers1 writers2;

            train_test_ratio2 = 0.8; % train test ratio
            percent_dataset2 = 0.02; % percentage of training and test data

        otherwise        
            error('Wrong dataset');

    end;

    file2.hist_indices = fullfile(Signsroot,'SavedData',dataset2,'hist_indices_compcorr.mat');
    file2.hists = fullfile(Signsroot,'SavedData',dataset2,'hists_compcorr.mat');

    nimages2 = size(all_images2,2);
    org_writers2 = unique(abs(all_writers2));
    norg_writers2 = length(org_writers2);

    %%%%%%%%%%%%  Prepare niter sets of test writers from dataset2  %%%%%%%%%%%

    test_writers = cell(1,niter);
    for i = 1:niter
        test_writers{i} = single(sort(randsample(1:norg_writers2,round((1-train_test_ratio2)*norg_writers2))));    
    end;
    ntest_writers = length(test_writers{1});

end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load and combine histograms with equal weights for dataset1

load(file1.hists,'histograms');

w = ones(size(histograms));        
w = w/sum(w);

for j = 1:size(histograms,1)
    histograms{j} = w(j)*histograms{j};
end;

histograms1 = cat(2,histograms{:});

clear histograms;
%% Load and combine histograms with equal weights for dataset2

load(file2.hists,'histograms');

w = ones(size(histograms));        
w = w/sum(w);

for j = 1:size(histograms,1)
    histograms{j} = w(j)*histograms{j};
end;

histograms2 = cat(2,histograms{:});

clear histograms;
%% Divide train and test set

Y1 = [];

for j = 1:ntrain_writers
    Y1 = [Y1;[ones(tot_sigs_pair11,1,'single');2*ones(tot_sigs_pair12,1,'single')]];
end;

Y2 = [];

for j = 1:ntest_writers
    Y2 = [Y2;[ones(tot_sigs_pair21,1,'single');2*ones(tot_sigs_pair22,1,'single')]];
end;

ntrain_set = round(percent_dataset1*min(histc(Y1, unique(Y1))));
ntest_set = round(percent_dataset2*min(histc(Y2, unique(Y2))));

accs = zeros(1,niter);
eers = zeros(1,niter);
fars = zeros(1,niter);
frrs = zeros(1,niter);

for iter = 1:niter
    
    fprintf('Iteration = %02d\n',iter);
    
    train_set = [];
    test_set = [];

    for j = 1:nclasses
        idx1 = find(Y1 == j);
        train_set = [train_set;sort(randsample(idx1,ntrain_set))];
        idx2 = find(Y2 == j);
        test_set = [test_set;sort(randsample(idx2,ntest_set))];

        clear idx1 idx2;
    end;

    Y_train = double(Y1(train_set,:));
    Y_test = double(Y2(test_set,:));
    vl_Y_test = Y_test';
    vl_Y_test(vl_Y_test==2) = -1;

    X1 = [];

    for j = 1:ntrain_writers

        idx1 = all_writers1 == train_writers{iter}(j);
        idx2 = all_writers1 == -train_writers{iter}(j);

        M1 = histograms1(idx1,:);
        M2 = histograms1(idx2,:);

        clear idx1 idx2;

        D11 = rowwise_couple_matrix(M1,M1);
        D11(logical(tril(ones(size(M1,1)))),:) = [];
        D12 = rowwise_couple_matrix(M1,M2);

        clear M1 M2;

        X1 = [X1;[D11;D12]];

        clear D11 D12;

    end;

    X2 = [];

    for j = 1:ntest_writers

        idx1 = all_writers2 == test_writers{iter}(j);
        idx2 = all_writers2 == -test_writers{iter}(j);

        M1 = histograms2(idx1,:);
        M2 = histograms2(idx2,:);

        clear idx1 idx2;

        D11 = rowwise_couple_matrix(M1,M1);
        D11(logical(tril(ones(size(M1,1)))),:) = [];
        D12 = rowwise_couple_matrix(M1,M2);

        clear M1 M2;

        X2 = [X2;[D11;D12]];

        clear D11 D12;

    end;

    X_train = X1(train_set,:);
    X_test = X2(test_set,:);

    clear X1 X2;
    
    % libsvm training and prediction

%     fprintf('Computing kernel for classification...');
% 
%     K_train = double([(1:size(X_train,1))' vl_alldist2(X_train',X_train',kernel)]);
%     K_test = double([(1:size(X_test,1))' vl_alldist2(X_test',X_train',kernel)]);
% 
%     clear X_train X_test;
% 
%     fprintf('Done.\n');
% 
%     best_model = 0;
% 
%     for j=1:length(Cs)    
%         options = sprintf(opt_str, Cs(j));
%         model = svmtrain(Y_train, K_train, options);
%         if(model>best_model)
%             best_model = model;
%             best_C = Cs(j);
%         end;
%     end;
% 
%     options = sprintf(strrep(opt_str,'-v 5 ',''),best_C);
% 
%     model_libsvm = svmtrain(Y_train,K_train,options);
% 
%     [~,acc,probs] = svmpredict(Y_test,K_test,model_libsvm,'-b 1');
    
    % liblinear training and prediction
    
    best_model = 0;
    best_C = NaN;
    
    % homogeneous kernel map
    X_train = vl_homkermap( X_train', 1, 'kernel', 'kinters', 'gamma', .5 )';
    X_test = vl_homkermap( X_test', 1, 'kernel', 'kinters', 'gamma', .5 )';
    
    X_train = sparse( double( X_train ) );
    X_test = sparse( double( X_test ) );
    Y_train = double( Y_train );
    Y_test = double( Y_test );

    for j=1:length(Cs)    
        options = sprintf( opt_str_liblinear, Cs(j) );
        model = train( Y_train, X_train, options );
        if( model > best_model )
            best_model = model;
            best_C = Cs(j);
        end;
    end;
    
    options = sprintf( strrep( opt_str_liblinear, '-v 5 ', '' ), best_C );

    model_liblinear = train( Y_train, X_train, options );
    
    [ Y_pred, acc, probs ] = predict( Y_test, X_test, model_liblinear, '-b 1' );

    scores = probs( :, model_liblinear.Label == 1 );

    [~, ~, info] = vl_roc( vl_Y_test, scores );
    
    far = nnz(Y_pred == 1 & Y_test == 2)/nnz(Y_test == 2);
    frr = nnz(Y_pred == 2 & Y_test == 1)/nnz(Y_test == 1);
    
    accs(iter) = acc(1);
    fars(iter) = far*100;
    frrs(iter) = frr*100;
    eers(iter) = info.eer*100;
    
end;

fprintf('Dataset = %s, Accuracy = %.2f, FAR = %.2f, FRR = %.2f, EER = %.2f\n\n',...
    dataset2, mean(accs), mean(fars), mean(frrs), mean(eers));
fp = fopen(fullfile(Signsroot,'Results',[dataset1,'_cross_dataset.txt']),'a');
fprintf(fp, 'Dataset = %s, Accuracy = %.2f, FAR = %.2f, FRR = %.2f, EER = %.2f\n\n',...
    dataset2, mean(accs), mean(fars), mean(frrs), mean(eers));
fclose(fp);