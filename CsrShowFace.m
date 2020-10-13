 function [img] = CsrShowFace(X,index,width,height)
     img = {};
     dimen = size(X,2);
     %------step1.重构图片-----------------------%
     for i = 1:dimen
        img{i} = reshape(X(:,i),width,height);
     end
     %保证显示图像最多12个
    index = min([index 12]);
     %------step2.显示图像-----------------------%
    for i = 1:index
        subplot(3,4,i);
        imshow(img{i});
    end
 
 end
 