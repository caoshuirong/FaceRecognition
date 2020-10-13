function  [w] = CsrPerceptron(varargin)
    C = nargin;%总的类别数
    rate = 1;%学习率
    dimen = size(varargin{1},1) + 1;%加一表示增广后的维数
    tmp = [];
 %---------step1.样本增广---------------%
    for i = 1:C
       tmp = ones(1,size(varargin{i},2));
       varargin{i} = [varargin{i}' tmp']';
    end

 %---------step2.初始化权向量------------%
    w = zeros(1,dimen);

%---------step3.对于两类情况迭代找系数---------------%
if nargin == 2
   varargin{2} = varargin{2}*(-1);%负样本*（-1）
   flag = 1;%是否需要继续迭代
   while flag
       flag = 0; %先设置为不需要迭代
       for i = 1:C %对每一类
           Xi = varargin{i};
           for j = 1:size(Xi,2)%每一类的样本数
                %tmp = w * Xi(:,j); %调试用
                if w * Xi(:,j) <= 0 %检查对每一类所有的样本是否判别正确
                    w = w + rate * Xi(:,j)' %判别错误则修正权值
                    flag = 1;%继续迭代
                end
           end
       end
   end

else%多模式类别
    %---------step2.初始化权向量------------%
    w = w';
    dVal = [];
    for i = 1:C-1
        tmp = zeros(dimen,1);
        w = [w zeros(dimen,1)];
    end
    %---------step3.迭代-------------------%
   flag = 1;%是否需要继续迭代
   while flag
       for i = 1:C %对每一类
           Xi = varargin{i};
           for j = 1:size(Xi,2)%每一个样本
               flag = 0; %先设置为不需要迭代
               sample = Xi(:,j);
               dVal = [];
                %计算样本与每一个判别函数的值
                for k = 1:C
                    dVal = [dVal w(:,k)'* sample]; 
                end
                
                %检查是否需要更新权重
                for k = 1:C
                    if dVal(i) <= dVal(k) && k ~= i  %肇事者减去
                        w(:,k) = w(:,k) - rate *sample;
                        flag = 1;
                    end
                end
                if flag == 1
                   w(:,i) = w(:,i) + rate *sample;%当事人加上
                end
                
           end
       end
   end
    
end


