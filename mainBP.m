load("Yale_15_11_100_80.mat")
Wih = [];Whj = [];
testX = [];
dimen = size(DAT,3);%��¼�������
%------step1.�ָ�ѵ�����Ͳ��Լ�-------------%
%ÿ����11��ͼƬ��ǰ���Ž���ѵ���������Ų���
DAT = DAT;
X = [];
for i = 1:dimen
    X = [X DAT(:,1:8,i)];
    testX = [testX DAT(:,9:11,i)];
end
%------step2.PCA��ά-----------------------------------%
[Y,eigVec,eigVal] = CsrPCA(X,size(X,2));%Y�ǽ�ά���ѵ������eigVec�������ռ�Ļ�����
testY = eigVec'*testX;%�����Լ��任����ά��������ռ�
Y = Y';testY = testY';%���������Ų�
bias = -1 * ones(size(Y,1),1);%ѵ����������ƫ��
Y = [Y bias];
bias = -1 * ones(size(testY,1),1);
testY = [testY bias];%������������ƫ��
%------step3.ѵ������һ������󽵵͵���������---------------%
Y = mapminmax(Y);
testY = mapminmax(testY);
%��ʼ��ѵ�������ı�ǩ
Tj = eye(15);
%------step4.��ʼ��Ȩ��-----------------------------------%
%rand("state",sum(100*clock));
hideLayerDimen = ceil(0.3*size(Y,2));%����ڵ����ǽ�ά������������30% 
outPutLayerDimen = 15;
Wih =  0.1 * rand(size(Y,2),hideLayerDimen);%��ʼȨ�ر���Сһ�㣬����ᱥ�ͣ��ݶ���ʧ
Whj =  0.1 * rand(hideLayerDimen + 1 ,outPutLayerDimen);% ���ز�Ľڵ��� + һ��bias
%------step4.������������ѵ��-----------------------------%
sample = [];count = 0;num = size(Y,1);
iterate = 0;%��������
learnRate = 0.15;%ѧϰ��
    error = 0.65*1e-3;
while(count < num && iterate < 10000)
  count = 0;%��ÿһ��������С�ڸ���error��ֹͣ����
  for i = 1:size(Y,1)
      sample = Y(i,:);
      [Wih,Whj,count] = CsrTranAnn(sample,Wih,Whj,Tj(:,ceil(i/8)),count,learnRate,error);
  end
  iterate = iterate + 1
end
%------step5.�����������в���--------------%
index = [0];
for i = 1:size(testY,1)
    index(i) = testBP(testY(i,:),Wih,Whj);
end
%------step6.������ȷ��---------------------%
accuracy = [0];
count = 0;
testNum = 3;%ÿ����Ĳ��Լ�������������
kinds = 15;%�������
for i = 1:kinds %15����
    for j = 1:testNum
        if index((i-1)*3+j) == i
           count = count +1; 
        end
    end
    accuracy(i) = count/testNum;
    count = 0;
end
res = sum(accuracy)/kinds








