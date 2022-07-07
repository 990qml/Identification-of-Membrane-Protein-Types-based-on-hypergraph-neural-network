function PSSM_Matrix = Read_Text_files_PSSM(file_name,Seq_Length)


%read file
fin=fopen(file_name,'r');

PSSM_Matrix = [];

n_l=zeros(20,1);
n_l=n_l';

str=fgetl(fin);	
ret = 0;
row_index = 1;
while feof(fin)==0;
	row_index = row_index +1;
	str=fgetl(fin);
	if ((row_index>=4)&(row_index<=Seq_Length+3))
		vect_v = [];
		A=strread(str,'%s','delimiter',' ');
		
		for j=3:22
			n_l(j-2) = str2num(cell2mat(A(j)));
		end
		
		vect_v = n_l;
		PSSM_Matrix = [PSSM_Matrix;vect_v];
		ret = ret + 1;
	end
		
	

end



fclose(fin);