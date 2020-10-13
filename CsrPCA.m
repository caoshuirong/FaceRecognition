function [Y,tmpVec,tmpVal] = CsrPCA(X,N)
    %预处理，将8000*11*15的数组合并成二维矩阵，便于求协方差矩阵
    X = reshape(X,8000,N);
    Y = [];
    %-----------step1.平均脸------------------------------%
    meanX = mean(X,2);
    imshow(reshape(meanX,100,80));
    %-----------step2.求协方差矩阵------------------------%
    C = (1/N) *( X * X') - (meanX * meanX');
    %-----------step3.求特征向量--------------------------%
    [vector,val] = eig(C);%求出对角阵特征值，并从小到大排列
    %-----------step4.排序-------------------------------%
    lambda = wrev(diag(val));%特征值去对角化，逆序排列
    vector = fliplr(vector);%矩阵左右镜像对称
    %-----------step5.选择主成分并显示--------------------%
    sumVal = 0;
    total = sum(lambda);
    index = [];
    for i = 1 : N
        sumVal = sumVal + lambda(i);
        if sumVal/total > 0.9   %一般取0.9，但由于每一类训练样本数只有八张，
                                %（8-1）小于取0.9时的维数（为31维），Bayes求每个类的协方差矩阵，会不满秩，无法求逆！   
           index = i;
            break;
        end
    end
    index
    tmpVal = lambda(1:index);
    tmpVec = vector(:,1:index)*50;%选择主成分，数值太小，放大50倍才看得到
    %显示特征脸
    CsrShowFace(tmpVec,index,100,80);
    %-----------step6.获得降维后的样本--------------------%
    tmpVec = tmpVec/norm(tmpVec);
    Y = tmpVec'* X;
end