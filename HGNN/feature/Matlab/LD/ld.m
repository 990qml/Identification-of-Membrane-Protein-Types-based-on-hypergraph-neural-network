function F = ld(protein)

F=[];
protein=cell2mat(protein);
%%%%%%%%%%%%%%%%%%%% amino acid grouping  氨基酸分组 7 大类
protein=regexprep(protein,'X',''); %%%% delete unknown proteins
protein=regexprep(protein,'A|G|V','a');
protein=regexprep(protein,'I|L|F|P','b');
protein=regexprep(protein,'C','c');
protein=regexprep(protein,'Y|M|T|S','d');
protein=regexprep(protein,'H|N|Q|W','e');
protein=regexprep(protein,'R|K','f');
protein=regexprep(protein,'D|E','g');

%%%%%%%%%%%%%%%%%%%
% 一条蛋白质划分成10个局部区域

L = length(protein); 
step=ceil(L/4); 

region_position=zeros(10,2);%存取序列起始位置

region_position(1,1)=1;
region_position(1,2)=step;

region_position(2,1)=1+step;
region_position(2,2)=step*2;


region_position(3,1)=1+2*step;
region_position(3,2)=step*3;


region_position(4,1)=1+3*step;
region_position(4,2)=L;


region_position(5,1)=1;
region_position(5,2)=step*2;


region_position(6,1)=1+2*step;
region_position(6,2)=L;

region_position(7,1)=1+step;
region_position(7,2)=step*3;


region_position(8,1)=1;
region_position(8,2)=step*3;



region_position(9,1)=1+step;
region_position(9,2)=L;



region_position(10,1)=1+ceil(step/2);
region_position(10,2)=L-ceil(step/2);

%%%%%%%%%%%%%%%%%%%
%每区域个编码子C、T、D

for i=1:10
	s_start = region_position(i,1);
	s_end = region_position(i,2);
	region_seq = protein(s_start:s_end);
	region_value = CTP_Features(region_seq,7);  %%%特征表示是一个63维的特征向量
	F=[F ,region_value];%一共630维，一个蛋白序列
end

F(find(isnan(F)))=0;
F(find(isinf(F)))=0;




