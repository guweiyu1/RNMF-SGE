function [TRAIN,class_Train,TEST,class_Test] = imread_PIE( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num)
%% 载入数据
% m_img = 56;   n_img = 46;                                                                %图像尺寸调整为m_img*n_img
% Train_num = 8;Test_num = 2;                                                              %每个对象选取8张图片组成训练集
% classnum=40;totlenuminclass=10;

%载入训练集到V，样本分类号到class_V,载入测试集到T,样本分类号到class_T
TRAIN = zeros(m_img * n_img , classnum * Train_num);                                 %原始数据矩阵,每一列为一张图片
class_Train = zeros(classnum * Train_num, 1);
TEST = zeros(m_img * n_img , classnum * Test_num);    
class_Test = zeros(classnum * Test_num, 1);

for i=1:classnum
    rN = randperm(totlenuminclass);                                                     %随机划分训练集和测试集
%     rN = 1:totlenuminclass;
    for j=1:Train_num
        img_path = ['E:\A数据库\数据集图片格式\PIE_32x32_files(backup)\' num2str((i-1)*totlenuminclass+rN(j))];
        img = imread(img_path,'jpg');
        img = imresize(img, [m_img n_img]);
%         img = coverd(img,6);
        img_vector = img(1 : m_img * n_img);                               %图片向量化，作为初始数据矩阵中的一个列
%         img_vector = imnoise(img_vector,'salt & pepper',0.1);
        TRAIN(:,j + (i - 1) * Train_num) = im2double(img_vector);                 %存储图片
        class_Train(j + (i - 1) * Train_num) = i;                              %存储类别信息
    end
    for j = 1 : Test_num
        img_path = ['E:\A数据库\数据集图片格式\PIE_32x32_files(backup)\' num2str((i-1)*totlenuminclass+rN(j+Train_num))];
        img = imread(img_path,'jpg');
        img = imresize(img, [m_img n_img]);
%         img = coverd(img,6);
        img_vector = img(1 : m_img * n_img);                               %图片向量化，作为初始数据矩阵中的一个列
%         img_vector = imnoise(img_vector,'salt & pepper',0.1);
        TEST(:,j + (i - 1) * Test_num) = im2double(img_vector);                  %存储图片
        class_Test(j + (i - 1) * Test_num) = i;                               %存储类别信息
    end
    
end