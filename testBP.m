function [index] = testBP(Xi,Wih,Whj)
        Yh = Xi * Wih;%���ز������
        Yh = CsrSigmoid(Yh);%���ز�����
        Yj = [Yh -1] * Whj;%����������
        Yj = CsrSigmoid(Yj);%���������
        [~ ,index] = max(Yj);%�ҵ���ֵ�����Ǹ����
end