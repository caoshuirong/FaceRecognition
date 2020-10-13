function [Y,tmpVec,tmpVal] = CsrPCA(X,N)
    %Ԥ������8000*11*15������ϲ��ɶ�ά���󣬱�����Э�������
    X = reshape(X,8000,N);
    Y = [];
    %-----------step1.ƽ����------------------------------%
    meanX = mean(X,2);
    imshow(reshape(meanX,100,80));
    %-----------step2.��Э�������------------------------%
    C = (1/N) *( X * X') - (meanX * meanX');
    %-----------step3.����������--------------------------%
    [vector,val] = eig(C);%����Խ�������ֵ������С��������
    %-----------step4.����-------------------------------%
    lambda = wrev(diag(val));%����ֵȥ�Խǻ�����������
    vector = fliplr(vector);%�������Ҿ���Գ�
    %-----------step5.ѡ�����ɷֲ���ʾ--------------------%
    sumVal = 0;
    total = sum(lambda);
    index = [];
    for i = 1 : N
        sumVal = sumVal + lambda(i);
        if sumVal/total > 0.9   %һ��ȡ0.9��������ÿһ��ѵ��������ֻ�а��ţ�
                                %��8-1��С��ȡ0.9ʱ��ά����Ϊ31ά����Bayes��ÿ�����Э������󣬻᲻���ȣ��޷����棡   
           index = i;
            break;
        end
    end
    index
    tmpVal = lambda(1:index);
    tmpVec = vector(:,1:index)*50;%ѡ�����ɷ֣���ֵ̫С���Ŵ�50���ſ��õ�
    %��ʾ������
    CsrShowFace(tmpVec,index,100,80);
    %-----------step6.��ý�ά�������--------------------%
    tmpVec = tmpVec/norm(tmpVec);
    Y = tmpVec'* X;
end