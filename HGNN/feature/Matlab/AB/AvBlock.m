function [ AC ] = AvBlock( P , n_divide)
%P is the given matrix representation 
%it does not work with protein with less than 20 amino-acids

n=size(P,1);
B=floor(n/n_divide);
AC=[];
for i=1:n_divide
    RID=P( (B*(i-1))+1:B*i,: );
    if min(size(RID))==1
        AC=[AC (RID)];
    elseif  min(size(RID))==0
        AC=[AC P(ceil(B*i),:)];
    else
        AC=[AC mean(RID)];
    end
end
AC(find(isnan(AC)))=0;
AC(find(isinf(AC)))=0;


end

