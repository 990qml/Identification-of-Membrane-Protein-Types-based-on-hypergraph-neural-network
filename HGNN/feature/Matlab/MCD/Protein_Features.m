function Matrix = Protein_Features(protein,n)

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
    eval(['Region_',num2str(j),'=','protein(region_p)',';']) 
    
    %num2str(n-j+1)      
end

 
regionaa = [];
bb = [];
for i = 1:2^n-2 %2^5-2 = 30
    Bin_num = dec2bin(i);   % 10000000 转成二进制
    len_Bin_num = length(Bin_num);  % 8 二进制长度
    Flag = 0;
       
    for ii = 1:len_Bin_num-1  % 如果序列中出现01则表示有不连续的区域；
        aa = i;
        Lett_Right = bitget(uint8(aa),ii); %取aa的二进制，从左到右第ii位置的数字
        Lett_Left  = bitget(uint8(aa),ii+1);
        if Lett_Left == 0 && Lett_Right == 1
           Flag = 1; %不连续区域
        end
    end
    
  if Flag ==0
       bb = [bb;str2num(dec2bin(i,n))];  %二进制长度为5的i
       regionaa = [];
    for j = 1:n
        CodeStr = dec2bin(i,n);
        if str2num(CodeStr(j)) == 1
           regionaa = [regionaa,eval(['Region_',num2str(j)])];
        end
    end
      
    if length(regionaa)~= 0
       region_value = CTP_Features(regionaa,7);  %%%特征表示是一个63维的特征向量
       Matrix = [Matrix,region_value];
    end
  end
end
            



