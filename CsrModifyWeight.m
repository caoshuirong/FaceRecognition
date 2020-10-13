function [Wih,Whj] = CsrModifyWeight(Wih,Whj,deltaJ,learnRate,Yh,Xi)
    deltaH = [];
    dWih = [0];dWhj = [0];
    %------step1.���������Ȩ�ظ�����-------------%
    for h = 1:size(Whj,1) 
        for j = 1:size(Whj,2)
            dWhj(h,j) = learnRate * Yh(h) * deltaJ(j);
        end
    end
    %------step2.�������ز������-----------------%
    for h = 1:size(Whj,1) - 1
        deltaH(h) = (Whj(h,:) * deltaJ'); %��Ȩ������
        deltaH(h) = deltaH(h) * Yh(h) * (1 - Yh(h));%����ÿ��������ÿ���ڵ�ĵ���ֵ
    end
    %------step3.�������ز�Ȩ�ظ�����-------------%
    for i = 1:size(Xi,2)
        for h = 1:size(Whj,1) - 1
             dWih(i,h) = learnRate * Xi(i) * deltaH(h);
        end
    end
    %------step4.����Ȩ��------------------------%
    Wih = Wih + dWih;
    Whj = Whj + dWhj;
end














