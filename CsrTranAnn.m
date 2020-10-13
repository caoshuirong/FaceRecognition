function [Wih,Whj,count] = CsrTranAnn(sample,Wih,Whj,Tj,count,learnRate,error)
    E = 0;
    Yh = [];Yj = [];
    deltaJ = [];
    %------step1.�����������-------------%
    Yh = sample * Wih;
    Yh = CsrSigmoid(Yh);%Yh = 1./(1 + exp(-Yh));����������滻
    Yh = [Yh -1];% ����-1����Ϊ������������bias
    %------step2.�����������-----------%
    Yj = Yh * Whj;
    Yj = CsrSigmoid(Yj);%Yj = 1./(1 + exp(-Yj));
    %------step3.������������-----------%
    deltaJ = Tj' - Yj;
    for i = 1:size(deltaJ,2)%���ƽ����
        E = E + (deltaJ(i))^2;
    end
    if E < 2 * error %�������������Ҫ���������һ��������ѵ��
        count = count + 1;
    else   %�������Ȩ��
        %��������������䣬�ŷ��Ϲ�ʽ�����ǣ����ӵ�����������Ϊԭ����1/30������ȷ�ʲ��䡣
        %deltaJ = deltaJ .* Yj .* (1-Yj);%!!!!!!!!!!!!!!ÿ���������������ÿ�����ĵ������ǵ�ˣ�����������
        [Wih,Whj] = CsrModifyWeight(Wih,Whj,deltaJ,learnRate,Yh,sample);
    end
end