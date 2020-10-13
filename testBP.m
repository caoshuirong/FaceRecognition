function [index] = testBP(Xi,Wih,Whj)
        Yh = Xi * Wih;%隐藏层的输入
        Yh = CsrSigmoid(Yh);%隐藏层的输出
        Yj = [Yh -1] * Whj;%输出层的输入
        Yj = CsrSigmoid(Yj);%输出层的输出
        [~ ,index] = max(Yj);%找到数值最大的那个类别
end