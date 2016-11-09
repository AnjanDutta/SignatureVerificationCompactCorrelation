function main_signature_verification_compcorr(dataset)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Paths %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% FIRST SET THE PATHS %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_libsvm = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/libsvm/matlab'; % path to matlab folder of libsvm
dir_liblinear = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/liblinear/matlab'; % path to matlab folder of liblinear
dir_vlfeat = '/backup/matlab/temp/pacharya/ANJAN/AdditionalTools/vlfeat'; % path to vlfeat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_clstrs = [200 500];% number of node and edge labels
nnode_descs_consd = 2e5; % number of node descriptors considered for kmeans
nedge_descs_consd = 4e5; % number of edge descriptors considered for kmeans
nclasses = 2; % number of classes of classification problem
kernel = 'KL1'; % kernel for classification
Cs = 2.^(-7:2:9); % SVM C parameter
opt_str_libsvm = '-s 0 -t 4 -v 5 -c %f -g 0.003 -b 1 -q'; % LIBSVM option string
opt_str_liblinear = '-s 0 -v 5 -c %f -e 0.0001 -q'; % LIBLINEAR option string
rng(0); % seeding randomization
max_chunk_size = 100; % chunk size in number of images
patchsize = 10; % patchsize of hog
cellsize = 8;
ncells = 1;
dim_feats = 36*ncells^2; % hog = 36, surf = 64
niter = 10;
% dataset = 'Hindi';
parts = strsplit(pwd, '/');
Signsroot = fullfile('/',parts{1:end-1}); % parent folder
se = strel('disk', 3, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%% Precomputed results %%%%%%%%%%%%%%%%%%%%%%%%%%%%

precomputed_histograms = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%% Force computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

force_compute.graphs = true;
force_compute.vocab = false;
force_compute.hist_indices = false;
force_compute.hists = false;
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
switch dataset
        
    case 'CEDAR' % CEDAR dataset        
        
        subdir1 = 'Datasets/CEDAR/full_org';
        subdir2 = 'Datasets/CEDAR/full_forg';
        
        % source files
        images1 = dir(fullfile(Signsroot,subdir1,'*.png'));
        writers1 = single(cellfun(@str2num,strtok(strrep(strrep({images1.name},'original_',''),'.png',''),'_')));
        images1 = cellfun(@(x)fullfile(Signsroot,subdir1,x),{images1.name},'UniformOutput',false);
        sigs_per_writer1 = min(histc(writers1, unique(writers1)));
        tot_sigs_pair1 = nchoosek(sigs_per_writer1,2);

        images2 = dir(fullfile(Signsroot,subdir2,'*.png'));
        writers2 = -single(cellfun(@str2num,strtok(strrep(strrep({images2.name},'forgeries_',''),'.png',''),'_')));
        images2 = cellfun(@(x)fullfile(Signsroot,subdir2,x),{images2.name},'UniformOutput',false);
        sigs_per_writer2 = min(histc(writers2, unique(writers2)));
        tot_sigs_pair2 = sigs_per_writer1*sigs_per_writer2;
        
        all_images = [images1,images2]; clear images1 images2;
        all_writers = [writers1,writers2]; clear writers1 writers2;

        train_test_ratio = 0.9; % train test ratio
        percent_dataset = 0.3; % percentage of training and test data
        
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
        tot_sigs_pair1 = nchoosek(sigs_per_writer1,2);
        
        images2 = images2{:};
        writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
        images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/GPDS300',x),images2,'UniformOutput',false);
        sigs_per_writer2 = min(histc(writers2, unique(writers2)));
        tot_sigs_pair2 = sigs_per_writer1*sigs_per_writer2;
        
        all_images = [images1;images2]'; clear images1 images2;
        all_writers = [writers1;writers2]; clear writers1 writers2;

        train_test_ratio = 0.5; % train test ratio
        percent_dataset = 0.02; % percentage of training and test data
        
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
        tot_sigs_pair1 = nchoosek(sigs_per_writer1,2);
        
        images2 = images2{:};
        writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
        images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Bengali',x),images2,'UniformOutput',false);
        sigs_per_writer2 = min(histc(writers2, unique(writers2)));
        tot_sigs_pair2 = sigs_per_writer1*sigs_per_writer2;
        
        all_images = [images1;images2]'; clear images1 images2;
        all_writers = [writers1;writers2]; clear writers1 writers2;
        
        train_test_ratio = 0.8; % train test ratio
        percent_dataset = 0.02; % percentage of training and test data
        
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
        tot_sigs_pair1 = nchoosek(sigs_per_writer1,2);
        
        images2 = images2{:};
        writers2 = -single(cellfun(@str2num,strtok(images2,'/')));
        images2 = cellfun(@(x)fullfile(Signsroot,'Datasets/Hindi',x),images2,'UniformOutput',false);
        sigs_per_writer2 = min(histc(writers2, unique(writers2)));
        tot_sigs_pair2 = sigs_per_writer1*sigs_per_writer2;
        
        all_images = [images1;images2]'; clear images1 images2;
        all_writers = [writers1;writers2]; clear writers1 writers2;
        
        train_test_ratio = 0.8; % train test ratio
        percent_dataset = 0.02; % percentage of training and test data
        
    otherwise        
        error('Wrong dataset');
        
end;

if( ~exist( fullfile( Signsroot,'SavedData',dataset ), 'dir' ) )
    mkdir( fullfile( Signsroot,'SavedData',dataset ) );
end;

file_vocabs = fullfile(Signsroot,'SavedData',dataset,'vocab_compcorr.mat');
file_hist_indices = fullfile(Signsroot,'SavedData',dataset,'hist_indices_compcorr.mat');
file_hists = fullfile(Signsroot,'SavedData',dataset,'hists_compcorr.mat');
file_kernels = fullfile(Signsroot,'SavedData',dataset,'kernels_compcorr.mat');
file_results = fullfile(Signsroot,'SavedData',dataset,'results_compcorr.txt');

nimages = size(all_images,2);
org_writers = unique(abs(all_writers));
norg_writers = length(org_writers);
nchunks = ceil(nimages/max_chunk_size);
nnode_descs_consd_perchunk = ceil(nnode_descs_consd/nchunks);
nedge_descs_consd_perchunk = ceil(nedge_descs_consd/nchunks);
chunk_sizes = single([ones(1,(nchunks - 1))*max_chunk_size nimages - (nchunks - 1)*max_chunk_size]);
coor_start = ceil(cellsize/2);

%%%%%%%%%%%%% Prepare niter sets of Training and Test writers %%%%%%%%%%%%%

train_writers = cell(1,niter);
test_writers = cell(1,niter);
for i = 1:niter
    train_writers{i} = single(sort(randsample(1:norg_writers,round(train_test_ratio*norg_writers))));    
    test_writers{i} = single(setdiff(1:norg_writers,train_writers{i}));    
end;
ntrain_writers = length(train_writers{1});
ntest_writers = length(test_writers{1});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Graph creation; shape context based

if(~precomputed_histograms)

    
    for ic = 1:nchunks
        
        file_graphs = fullfile(Signsroot,'SavedData',dataset,sprintf('graphs_compcorr_%03d.mat',ic));

        if(~exist(file_graphs,'file')||force_compute.graphs)

            V = cell(chunk_sizes(ic),1);
            nV = zeros(chunk_sizes(ic),1,'single');
            descV = cell(chunk_sizes(ic),1);
            E = cell(chunk_sizes(ic),1);
            nE = zeros(chunk_sizes(ic),1,'single');
            descE = cell(chunk_sizes(ic),1);

            for is = 1:chunk_sizes(ic)

                if(ic == 1)        
                    iim = is;            
                else            
                    iim = (ic-1)*chunk_sizes(ic-1) + is;            
                end;

                fprintf('Current image=%d. ',iim);

                tic;

                im = im2single(imread(all_images{iim}));
                imsz = size(im);
                if(size(im,3)==3)
                    im = rgb2gray(im);
                end;
                
%%%%%%%%%%%%%%%%%%%%% grid points on signature pixels %%%%%%%%%%%%%%%%%%%%%
                
                mask = imdilate( ~imbinarize( im ), se );
                [ycoors_mask, xcoors_mask] = find( mask );
                vertices_mask = single([xcoors_mask ycoors_mask]);
                clear xcoors_mask ycoors_mask;
                
                xcoor_end = imsz(2);
                ycoor_end = imsz(1);    
                [xcoors_grid, ycoors_grid] = meshgrid(coor_start:...
                cellsize:xcoor_end,coor_start:cellsize:ycoor_end);                
                vertices_grid = single([xcoors_grid(:) ycoors_grid(:)]);
                clear xcoors_grid ycoors_grid;
                
                [vertices, ~, idx] = intersect( vertices_mask, vertices_grid, 'rows' );
                clear vertices_mask vertices_grid;

                descs = vl_hog(im,cellsize,'variant','dalaltriggs');
                descs = reshape(descs,[],dim_feats);
                descs = descs( idx, : );
                clear im idx;
                
                % nearest nbrs
                dist = pdist(vertices);
                dist(~dist) = Inf;
                [I,J] = find(squareform(dist<=2*cellsize));
                clear dist;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% grid points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                 descs = vl_hog(im,cellsize,'variant','dalaltriggs');
%                 descs = reshape(descs,[],dim_feats);            
%                 clear im;
%     
%                 xcoor_end = imsz(2);
%                 ycoor_end = imsz(1);
%     
%                 [xcoors,ycoors] = meshgrid(coor_start:cellsize:xcoor_end,...
%                     coor_start:cellsize:ycoor_end);
%                 
%                 vertices = single([xcoors(:) ycoors(:)]);
%                 clear xcoors ycoors;
%                 
%                 % nearest nbrs
%                 dist = pdist(vertices);
%                 dist(~dist) = Inf;
%                 [I,J] = find(squareform(dist<=2*cellsize));
%                 clear dist;
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% feature points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                 points = detectHarrisFeatures(im);
%                 points = detectBRISKFeatures(im);
%                 [descs,points] = my_vlhogptws(im,points,patchsize,ncells);
%                 clear im;                        
%                 vertices = points.Location;
%                 clear points;
% 
%                 % delaunay triangulation            
%                 tri = delaunay(double(vertices(:,1)),double(vertices(:,2)));
%                 e = [tri(:,[1 2]);tri(:,[2 3]);tri(:,[3 1])];
%                 edges = sparse(e(:,1),e(:,2),ones(size(e,1),1));clear e;
%                 edges = edges|edges';
%                 [I,J] = find(edges);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                V{is} = vertices;
                nV(is) = size(vertices,1);
                descV{is} = descs;
                descV{is} = bsxfun(@times, descV{is},1./(sum(descV{is},2)+eps));            
                clear vertices;

                E{is} = single([I,J]);
                nE(is) = size(E{is},1);
                descE{is} = [descs(I,:) descs(J,:)];
                descE{is} = bsxfun(@times, descE{is},1./(sum(descE{is},2)+eps));            
                clear descs I J;

                G.V = V{is};
                G.E = E{is};

                toc;

            end;
            fprintf('Saving chunk: %04d...',ic);
            save(file_graphs,'V','nV','descV','E','nE','descE','-v7.3');
            fprintf('Done.\n');
            clear V nV descV E nE descE;

        end;

    end;

    %% Vocabulary

    if(~exist(file_vocabs,'file')||force_compute.vocab)

        arr_descV = zeros(nnode_descs_consd_perchunk*nchunks,dim_feats,'single'); % need to put the dimension
        arr_descE = zeros(nedge_descs_consd_perchunk*nchunks,2*dim_feats,'single'); % need to put the dimension

        for ic = 1:nchunks

            file_graphs = fullfile(Signsroot,'SavedData',dataset,sprintf('graphs_compcorr_%03d.mat',ic));

            if(~exist(file_graphs,'file'))
                error('Error: %s does not exist.\n',file_graphs);
            else

                fprintf('Loading from chunk: %03d...',ic);
                load(file_graphs,'descV','descE');

                %%%%%%%%%%%% nodes %%%%%%%%%%%

                descVi = cat(1,descV{:});
                clear descV1;
                sz = size(descVi,1);
                if(nnode_descs_consd_perchunk<sz)
                    idx1 = sort(randsample(sz,nnode_descs_consd_perchunk));
                    descVi = descVi(idx1,:);
                    sz = nnode_descs_consd_perchunk;
                    clear idx1;
                end;            
                str_idx = (ic-1)*nnode_descs_consd_perchunk+1;
                end_idx = str_idx+sz-1;
                arr_descV(str_idx:end_idx,:) = descVi;
                arr_descV(end_idx+1:end,:) = [];
                clear str_idx end_idx descVi;

                %%%%%%%%%%%% edges %%%%%%%%%%%

                descEi = cat(1,descE{:});
                clear descE1;
                sz = size(descEi,1);
                if(nedge_descs_consd_perchunk<sz)
                    idx1 = sort(randsample(sz,nedge_descs_consd_perchunk));
                    descEi = descEi(idx1,:);
                    sz = nedge_descs_consd_perchunk;
                    clear idx1;
                end;
                str_idx = (ic-1)*nedge_descs_consd_perchunk+1;
                end_idx = str_idx+sz-1;
                arr_descE(str_idx:end_idx,:) = descEi;
                arr_descE(end_idx+1:end,:) = [];
                clear str_idx end_idx descEi;

                fprintf('Done.\n');

            end;

        end;

        fprintf('Creating node vocabulary...');

        descs = vl_colsubset(arr_descV',nnode_descs_consd);
        [node_cluster_cntrs,~] = vl_kmeans(descs,num_clstrs(1),'Initialization','plusplus','Algorithm','elkan');
        node_cluster_cntrs = node_cluster_cntrs';

        save(file_vocabs,'node_cluster_cntrs');

        clear descs arr_descV node_cluster_cntrs;

        fprintf('Done.\n');

        fprintf('Creating edge vocabulary...');

        desc_pairs = vl_colsubset(arr_descE',nedge_descs_consd);
        [edge_cluster_cntrs,~] = vl_kmeans(desc_pairs,num_clstrs(2),'Initialization','plusplus','Algorithm','elkan');
        edge_cluster_cntrs = edge_cluster_cntrs';

        save(file_vocabs,'edge_cluster_cntrs','-append');

        clear desc_pairs arr_descE edge_cluster_cntrs;

        fprintf('Done.\n');

    end;

    %% Assigning labels

    vars = whos('-file',file_graphs);
    if(~(ismember('LV', {vars.name})||ismember('LE', {vars.name}))||force_compute.vocab)

        fprintf('Loading vocabularies...');
        load(file_vocabs,'node_cluster_cntrs','edge_cluster_cntrs');
        fprintf('Done.\n');

        for ic = 1:nchunks

            file_graphs = fullfile(Signsroot,'SavedData',dataset,sprintf('graphs_compcorr_%03d.mat',ic));

            fprintf('Updating chunk: %03d...',ic);

            load(file_graphs,'descV','descE');

            LV = cellfun(@knnsearch,repmat({node_cluster_cntrs},chunk_sizes(ic),1),descV,'UniformOutput',false);
            LE = cellfun(@knnsearch,repmat({edge_cluster_cntrs},chunk_sizes(ic),1),descE,'UniformOutput',false);

            save(file_graphs,'LV','LE','-append');

            clear descV descE LV LE;

            fprintf('Done.\n');

        end;

        clear node_cluster_cntrs edge_cluster_cntrs;

    end;

    %% Histogram indices

    if(~exist(file_hist_indices,'file')||force_compute.hist_indices)

        idx_image = cell(length(num_clstrs),1);
        idx_bin = cell(length(num_clstrs),1);

        for ic = 1:nchunks

            file_graphs = fullfile(Signsroot,'SavedData',dataset,sprintf('graphs_compcorr_%03d.mat',ic));

            if(~exist(file_graphs,'file'))
                error('Error: %s does not exist.\n',file_graphs);
            else

                load(file_graphs,'LV','LE');

                for is = 1:chunk_sizes(ic)

                    if(ic == 1)        
                        iim = is;            
                    else            
                        iim = (ic-1)*chunk_sizes(ic-1) + is;            
                    end;            

                    fprintf('Computing histogram indices for image: %d. ',iim);

                    tic;

                    idx_image{1} = [idx_image{1,1};iim*ones(size(LV{is}),'single')];
                    idx_bin{1} = [idx_bin{1};LV{is}];

                    idx_image{2} = [idx_image{2};iim*ones(size(LE{is}),'single')];
                    idx_bin{2} = [idx_bin{2};LE{is}];

                    toc;

                end;

                clear LV LE;

            end;

        end;

        save(file_hist_indices,'idx_image','idx_bin','num_clstrs','-v7.3');
        clear idx_image idx_bin num_clstrs;

    end;

    %% Compute histograms

    if(~exist(file_hists,'file')||force_compute.hists)

        load(file_hist_indices,'idx_image','idx_bin','num_clstrs');

        histograms = cell(length(num_clstrs),2);

        fprintf('Computing histograms...');

        for i = 1:length(num_clstrs)

            histograms{i} = zeros(nimages,num_clstrs(i),'single');
            histograms{i} = vl_binsum(histograms{i},single(1),sub2ind([nimages,num_clstrs(i)],idx_image{i},idx_bin{i}));
            histograms{i} = bsxfun(@times,histograms{i},1./(sum(histograms{i},2)+eps));

        end;        

        fprintf('Done.\n');

        save(file_hists,'histograms','-v7.3');    
        clear idx_image idx_bin num_clstrs histograms;

    end;
    
end;

%% Load and combine histograms with equal weights

load(file_hists,'histograms');

w = ones(size(histograms));        
w = w/sum(w);

for j = 1:size(histograms,1)
    histograms{j} = w(j)*histograms{j};
end;

histogram = cat(2,histograms{:});

%% Divide train and test set

Y1 = [];

for j = 1:ntrain_writers
    Y1 = [Y1;[ones(tot_sigs_pair1,1,'single');2*ones(tot_sigs_pair2,1,'single')]];
end;

Y2 = [];

for j = 1:ntest_writers
    Y2 = [Y2;[ones(tot_sigs_pair1,1,'single');2*ones(tot_sigs_pair2,1,'single')]];
end;

ntrain_set = round(percent_dataset*min(histc(Y1, unique(Y1))));
ntest_set = round(percent_dataset*min(histc(Y2, unique(Y2))));

accs = zeros(1,niter);
eers = zeros(1,niter);

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

        idx1 = all_writers == train_writers{iter}(j);
        idx2 = all_writers == -train_writers{iter}(j);

        M1 = histogram(idx1,:);
        M2 = histogram(idx2,:);

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

        idx1 = all_writers == test_writers{iter}(j);
        idx2 = all_writers == -test_writers{iter}(j);

        M1 = histogram(idx1,:);
        M2 = histogram(idx2,:);

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
    X_train = vl_homkermap( X_train', 3, 'kernel', 'kinters' )';
    X_test = vl_homkermap( X_test', 3, 'kernel', 'kinters' )';
    
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
    
    [ ~, acc, probs ] = predict( Y_test, X_test, model_liblinear, '-b 1' );

    scores = probs( :, model_liblinear.Label == 1 );

    [~,~,info] = vl_roc( vl_Y_test, scores );
    
    accs(iter) = acc(1);
    eers(iter) = info.eer*100;
    
end;

fprintf('Final Accuracy = %.2f, EER = %.2f\n\n',mean(accs),mean(eers));
