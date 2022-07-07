N_A_feature=[];
N_B_feature=[];
P_A_feature=[];
P_B_feature=[];
%load('Matine_Data.mat');
load('P_protein_a.mat');
load('P_protein_b.mat');
load('N_protein_b.mat');
load('N_protein_a.mat');

%N_A
for i=1:size(N_protein_a)
    SEQ=N_protein_a(i);
	FF=ld(SEQ);
    N_A_feature(i,:)=FF;
end

%N_B
for i=1:size(N_protein_b)
    SEQ=N_protein_b(i);
	FF=ld(SEQ);
    N_B_feature(i,:)=FF;
end

%P_A

for i=1:size(P_protein_a)
    SEQ=P_protein_a(i);
	FF=ld(SEQ);
    P_A_feature(i,:)=FF;
end


%P_B
for i=1:size(P_protein_b)
    SEQ=P_protein_b(i);
	FF=ld(SEQ);
    P_B_feature(i,:)=FF;
end

P_LD = [P_A_feature,P_B_feature];
N_LD = [N_A_feature,N_B_feature];
LD = [P_LD;N_LD];