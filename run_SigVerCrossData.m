datasets = {'CEDAR', 'Bengali', 'Hindi', 'GPDS300', 'GPDS960'};
parfor i = 1:length(datasets)
    for j = 1:length(datasets)
        main_signature_verification_compcorr_cross_dataset(datasets{i},datasets{j})
    end;
end;