function [Wih,Whj] = CsrModifyWeight(Wih,Whj,deltaJ,learnRate,Yh,Xi)
    deltaH = [];
    dWih = [0];dWhj = [0];
    %------step1.计算输出层权重更新量-------------%
    for h = 1:size(Whj,1) 
        for j = 1:size(Whj,2)
            dWhj(h,j) = learnRate * Yh(h) * deltaJ(j);
        end
    end
    %------step2.计算隐藏层总误差-----------------%
    for h = 1:size(Whj,1) - 1
        deltaH(h) = (Whj(h,:) * deltaJ'); %加权输出误差
        deltaH(h) = deltaH(h) * Yh(h) * (1 - Yh(h));%总误差，每个结点乘以每个节点的导数值
    end
    %------step3.计算隐藏层权重更新量-------------%
    for i = 1:size(Xi,2)
        for h = 1:size(Whj,1) - 1
             dWih(i,h) = learnRate * Xi(i) * deltaH(h);
        end
    end
    %------step4.更新权重------------------------%
    Wih = Wih + dWih;
    Whj = Whj + dWhj;
end














