function  [w] = CsrPerceptron(varargin)
    C = nargin;%�ܵ������
    rate = 1;%ѧϰ��
    dimen = size(varargin{1},1) + 1;%��һ��ʾ������ά��
    tmp = [];
 %---------step1.��������---------------%
    for i = 1:C
       tmp = ones(1,size(varargin{i},2));
       varargin{i} = [varargin{i}' tmp']';
    end

 %---------step2.��ʼ��Ȩ����------------%
    w = zeros(1,dimen);

%---------step3.�����������������ϵ��---------------%
if nargin == 2
   varargin{2} = varargin{2}*(-1);%������*��-1��
   flag = 1;%�Ƿ���Ҫ��������
   while flag
       flag = 0; %������Ϊ����Ҫ����
       for i = 1:C %��ÿһ��
           Xi = varargin{i};
           for j = 1:size(Xi,2)%ÿһ���������
                %tmp = w * Xi(:,j); %������
                if w * Xi(:,j) <= 0 %����ÿһ�����е������Ƿ��б���ȷ
                    w = w + rate * Xi(:,j)' %�б����������Ȩֵ
                    flag = 1;%��������
                end
           end
       end
   end

else%��ģʽ���
    %---------step2.��ʼ��Ȩ����------------%
    w = w';
    dVal = [];
    for i = 1:C-1
        tmp = zeros(dimen,1);
        w = [w zeros(dimen,1)];
    end
    %---------step3.����-------------------%
   flag = 1;%�Ƿ���Ҫ��������
   while flag
       for i = 1:C %��ÿһ��
           Xi = varargin{i};
           for j = 1:size(Xi,2)%ÿһ������
               flag = 0; %������Ϊ����Ҫ����
               sample = Xi(:,j);
               dVal = [];
                %����������ÿһ���б�����ֵ
                for k = 1:C
                    dVal = [dVal w(:,k)'* sample]; 
                end
                
                %����Ƿ���Ҫ����Ȩ��
                for k = 1:C
                    if dVal(i) <= dVal(k) && k ~= i  %�����߼�ȥ
                        w(:,k) = w(:,k) - rate *sample;
                        flag = 1;
                    end
                end
                if flag == 1
                   w(:,i) = w(:,i) + rate *sample;%�����˼���
                end
                
           end
       end
   end
    
end


