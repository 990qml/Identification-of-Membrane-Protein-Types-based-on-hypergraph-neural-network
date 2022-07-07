function [ AC ] = AutoCov( P, lg )
%P is the protein
%lg is the lag, it denotes the distance between one residue and its neighbors

n=size(P,1);
clear AC;
type_n = size(P,2);
for lag=1:lg
    for j=1:type_n
        A=1/(n-lag);
        AC(lag,j)=0;
        for i=1:n-lag
            AC(lag,j)=AC(lag,j)+(P(i,j)-(1/n)*sum(P(:,j)))*(P(i+lag,j)-(1/n)*sum(P(:,j)));
        end
        AC(lag,j)=A*AC(lag,j);
    end
end
AC(find(isnan(AC)))=0;
AC(find(isinf(AC)))=0;
AC=single(AC(:));

