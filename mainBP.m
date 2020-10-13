load("Yale_15_11_100_80.mat")
Wih = [];Whj = [];
testX = [];
dimen = size(DAT,3);%记录总类别数
%------step1.分割训练集和测试集-------------%
%每个人11张图片，前八张进行训练，后三张测试
DAT = DAT;
X = [];
for i = 1:dimen
    X = [X DAT(:,1:8,i)];
    testX = [testX DAT(:,9:11,i)];
end
%------step2.PCA降维-----------------------------------%
[Y,eigVec,eigVal] = CsrPCA(X,size(X,2));%Y是降维后的训练集，eigVec是特征空间的基向量
testY = eigVec'*testX;%将测试集变换到降维后的特征空间
Y = Y';testY = testY';%样本按行排布
bias = -1 * ones(size(Y,1),1);%训练样本增加偏置
Y = [Y bias];
bias = -1 * ones(size(testY,1),1);
testY = [testY bias];%测试样本增加偏置
%------step3.训练集归一化，大大降低迭代次数！---------------%
Y = mapminmax(Y);
testY = mapminmax(testY);
%初始化训练样本的标签
Tj = eye(15);
%------step4.初始化权重-----------------------------------%
%rand("state",sum(100*clock));
hideLayerDimen = ceil(0.3*size(Y,2));%隐层节点数是降维后总像素数的30% 
outPutLayerDimen = 15;
Wih =  0.1 * rand(size(Y,2),hideLayerDimen);%初始权重必须小一点，否则会饱和，梯度消失
Whj =  0.1 * rand(hideLayerDimen + 1 ,outPutLayerDimen);% 隐藏层的节点数 + 一个bias
%------step4.输入样本进行训练-----------------------------%
sample = [];count = 0;num = size(Y,1);
iterate = 0;%迭代次数
learnRate = 0.15;%学习率
    error = 0.65*1e-3;
while(count < num && iterate < 10000)
  count = 0;%当每一个样本都小于给定error，停止迭代
  for i = 1:size(Y,1)
      sample = Y(i,:);
      [Wih,Whj,count] = CsrTranAnn(sample,Wih,Whj,Tj(:,ceil(i/8)),count,learnRate,error);
  end
  iterate = iterate + 1
end
%------step5.输入样本进行测试--------------%
index = [0];
for i = 1:size(testY,1)
    index(i) = testBP(testY(i,:),Wih,Whj);
end
%------step6.计算正确率---------------------%
accuracy = [0];
count = 0;
testNum = 3;%每个类的测试集是连续的三张
kinds = 15;%总类别数
for i = 1:kinds %15个人
    for j = 1:testNum
        if index((i-1)*3+j) == i
           count = count +1; 
        end
    end
    accuracy(i) = count/testNum;
    count = 0;
end
res = sum(accuracy)/kinds








