function D = rowwise_couple_matrix(M1,M2)

nM1 = size(M1,1);
nM2 = size(M2,1);
D = abs(repmat(M1,nM2,1)-kron(M2,ones(nM1,1)));

end