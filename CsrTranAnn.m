function [Wih,Whj,count] = CsrTranAnn(sample,Wih,Whj,Tj,count,learnRate,error)
    E = 0;
    Yh = [];Yj = [];
    deltaJ = [];
    %------step1.计算隐层输出-------------%
    Yh = sample * Wih;
    Yh = CsrSigmoid(Yh);%Yh = 1./(1 + exp(-Yh));激活函数，可替换
    Yh = [Yh -1];% 加上-1，是为了增加输出层的bias
    %------step2.计算输出层结果-----------%
    Yj = Yh * Whj;
    Yj = CsrSigmoid(Yj);%Yj = 1./(1 + exp(-Yj));
    %------step3.计算输出层误差-----------%
    deltaJ = Tj' - Yj;
    for i = 1:size(deltaJ,2)%误差平方和
        E = E + (deltaJ(i))^2;
    end
    if E < 2 * error %该样本误差满足要求，则进行下一个样本的训练
        count = count + 1;
    else   %否则更新权重
        %加上下面这条语句，才符合公式。但是，不加迭代次数减少为原来的1/30，且正确率不变。
        %deltaJ = deltaJ .* Yj .* (1-Yj);%!!!!!!!!!!!!!!每个结点的输出误差乘以每个结点的导数，是点乘！！！！！！
        [Wih,Whj] = CsrModifyWeight(Wih,Whj,deltaJ,learnRate,Yh,sample);
    end
end