function Matrix = fun(protein,n)

%2014-02-11

% protein is a protein sequence form a protein pair
Matrix=[];
AAindex = 'ACDEFGHIKLMNPQRSTVWY';

%%%%%%%%%%%%%%%%%%%% amino acid grouping  氨基酸分组
protein=regexprep(protein,'X',''); %%%% delete unknown proteins
protein=regexprep(protein,'A|G|V','a');
protein=regexprep(protein,'I|L|F|P','b');
protein=regexprep(protein,'C','c');
protein=regexprep(protein,'Y|M|T|S','d');
protein=regexprep(protein,'H|N|Q|W','e');
protein=regexprep(protein,'R|K','f');
protein=regexprep(protein,'D|E','g');
%%%%%%%%%%%%%%%%%%%% 一条蛋白质划分成2^n-1个局部区域，对每个局部区域计算3个编码子C、T、D

L = length(protein); 
step=ceil(L/n); %向上取整 ceil(61/5)=13 n=5 
for j=1:n   %  将protein sequence分成n等分；1000: Region_1;   Region_2;   Region_3;   Region_4;
    if j~=n %不等于
        st=(j-1)*step+1; %1 14 27 40 
		
        sed=j*step; %13 26 39 52
    else
        st=(j-1)*step+1; %53
        sed=L; %61
    end
    region_p=[st:sed]; %%%% region position
    eval(['Region_',num2str(j),'=','protein(region_p)',';'])  %num2str(n-j+1)      
end
Matrix = protein