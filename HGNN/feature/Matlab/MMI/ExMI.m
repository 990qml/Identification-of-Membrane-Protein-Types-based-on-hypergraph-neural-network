function [feature_MMI] = ExMI(protein_A_sequence_f_1,protein_B_sequence_f_1,mmi_map)


feature_MMI = [];
P_feature=[];
N_feature=[];



%P_A

for i=1:size(protein_A_sequence_f_1)
    SEQ=protein_A_sequence_f_1(i);
	F = MMI(SEQ,mmi_map);
    P_feature(i,:)=F;
	kd = mod(i,100);
	if kd==0
		prin = i;
		prin
	end
end


%P_B
for i=1:size(protein_B_sequence_f_1)
    SEQ=protein_B_sequence_f_1(i);
	F = MMI(SEQ,mmi_map);
    N_feature(i,:)=F;
	kd = mod(i,100);
	if kd==0
		prin = i;
		prin
	end
end

feature_MMI = [P_feature;N_feature];



