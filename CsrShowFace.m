 function [img] = CsrShowFace(X,index,width,height)
     img = {};
     dimen = size(X,2);
     %------step1.�ع�ͼƬ-----------------------%
     for i = 1:dimen
        img{i} = reshape(X(:,i),width,height);
     end
     %��֤��ʾͼ�����12��
    index = min([index 12]);
     %------step2.��ʾͼ��-----------------------%
    for i = 1:index
        subplot(3,4,i);
        imshow(img{i});
    end
 
 end
 