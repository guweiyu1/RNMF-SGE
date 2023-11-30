function [B] = coverd(B,nn)
%加遮挡
[m,n] = size(B);
if nn ~= 0
    a = randperm(m-nn,1);
%     a = 30;
    b = randperm(n-nn,1);
%     b = 30;
    for x1 = a:a+nn
       for y1 = b:b+nn
          B(x1,y1) = 0;
       end
    end
else
    B = B;
end
end