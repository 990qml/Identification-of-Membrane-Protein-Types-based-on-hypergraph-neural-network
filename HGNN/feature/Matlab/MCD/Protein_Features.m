function Matrix = Protein_Features(protein,n)

%2014-02-11

% protein is a protein sequence form a protein pair
Matrix=[];
AAindex = 'ACDEFGHIKLMNPQRSTVWY';

%%%%%%%%%%%%%%%%%%%% amino acid grouping  ���������
protein=regexprep(protein,'X',''); %%%% delete unknown proteins
protein=regexprep(protein,'A|G|V','a');
protein=regexprep(protein,'I|L|F|P','b');
protein=regexprep(protein,'C','c');
protein=regexprep(protein,'Y|M|T|S','d');
protein=regexprep(protein,'H|N|Q|W','e');
protein=regexprep(protein,'R|K','f');
protein=regexprep(protein,'D|E','g');
%%%%%%%%%%%%%%%%%%%% һ�������ʻ��ֳ�2^n-1���ֲ����򣬶�ÿ���ֲ��������3��������C��T��D

L = length(protein); 
step=ceil(L/n); %����ȡ�� ceil(61/5)=13 n=5 
for j=1:n   %  ��protein sequence�ֳ�n�ȷ֣�1000: Region_1;   Region_2;   Region_3;   Region_4;
    if j~=n %������
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
    Bin_num = dec2bin(i);   % 10000000 ת�ɶ�����
    len_Bin_num = length(Bin_num);  % 8 �����Ƴ���
    Flag = 0;
       
    for ii = 1:len_Bin_num-1  % ��������г���01���ʾ�в�����������
        aa = i;
        Lett_Right = bitget(uint8(aa),ii); %ȡaa�Ķ����ƣ������ҵ�iiλ�õ�����
        Lett_Left  = bitget(uint8(aa),ii+1);
        if Lett_Left == 0 && Lett_Right == 1
           Flag = 1; %����������
        end
    end
    
  if Flag ==0
       bb = [bb;str2num(dec2bin(i,n))];  %�����Ƴ���Ϊ5��i
       regionaa = [];
    for j = 1:n
        CodeStr = dec2bin(i,n);
        if str2num(CodeStr(j)) == 1
           regionaa = [regionaa,eval(['Region_',num2str(j)])];
        end
    end
      
    if length(regionaa)~= 0
       region_value = CTP_Features(regionaa,7);  %%%������ʾ��һ��63ά����������
       Matrix = [Matrix,region_value];
    end
  end
end
            



