clear all; tic; NN= 1 ; acc=ones(1,NN)*nan;
for j=1 : NN
i = 60;
classnum = 68;totlenuminclass = 170;Train_num = i;Test_num = totlenuminclass-Train_num;    %每个对象选取8张图片组成训练集
m_img = 32;   n_img = 32;                                                                %图像尺寸调整为m_img*n_img
% [TRAIN,class_Train,TEST,class_Test] = imread_ORL( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num);
% [TRAIN,class_Train,TEST,class_Test] = imread_Yaleb( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num); %32*28
% [TRAIN,class_Train,TEST,class_Test] = imread_MNIST( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num); %20*20
[TRAIN,class_Train,TEST,class_Test] = imread_PIE( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num); %46*46
% [TRAIN,class_Train,TEST,class_Test] = imread_AR( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num);
% [TRAIN,class_Train,TEST,class_Test] = imread_coil( m_img, n_img, classnum, totlenuminclass, Train_num, Test_num);
%%
% [W1,H,Ht,class_recT1,rc] = NMF_mse( TRAIN, class_Train, m_img, n_img, Train_num, ...
%                     15, 50, TEST, class_Test, Test_num, classnum, totlenuminclass);
% [Ug,V2,Vt2,class_recT2,rc] = GNMF_mse( TRAIN, class_Train, Train_num, ...
%                         1,20, 50, TEST, class_Test, Test_num, classnum, totlenuminclass);
% [Uc,Vc,Vct,class_recT,rc] = CNMF_mse( TRAIN, class_Train, m_img, n_img, Train_num, ...
%                      20, 50, TEST, class_Test, Test_num, classnum, totlenuminclass);
% [Urs,Vrs,Vrst,class_recT,rc] = RSNMF_mse( TRAIN, class_Train, m_img, n_img, Train_num, ...
%                       20/classnum, 100, 0.1, TEST, class_Test, Test_num, classnum, totlenuminclass);
% [Udc,Vdc,Vdct,class_recT,rc] = NMFDC_mse( TRAIN, class_Train, m_img, n_img, Train_num, ...
%                       20, 50, 0.1, TEST, class_Test, Test_num, classnum, totlenuminclass);
% [Uan,Van,Vtan,class_recT,rc] = NMFAN_mse( TRAIN, class_Train, m_img, n_img, Train_num,...
%                     20, i, 1, 50, TEST, class_Test, Test_num, classnum, totlenuminclass);%r，近邻，λ，次数，
% [Uag,Vag,Vtag,class_recT,rc] = NMF_LCAG_mse( TRAIN, class_Train, m_img, n_img, Train_num,...
%                     20, i, 0.01, 0.01, 0.01, 0.01, 50, TEST, class_Test, Test_num, classnum, totlenuminclass);
[U,W,Y,Vt,class_recT,rc,J] = RNMF_SGE_mse( TRAIN, class_Train, m_img, n_img, Train_num,... 
                    68, 1000, 0.1, TEST, class_Test, Test_num, classnum, totlenuminclass);  WW = full(W);
acc(j)=rc;             %不同数据集，西格玛要重设，不同训练样本，迭代次数要重设，要多次迭代选最优。
end
toc;fprintf(2,'MAX & MIN = %.2f  &  %.2f ,  AVG & STD = %.2f±%.2f\n',max(acc)*100,min(acc)*100,mean(acc)*100,std(acc)*100);
%% display(mean(acc))
% acc=[nan,0.8222,0.8583,0.9619,0.9556,nan];
% XX1=U*(W*Y)';
% V=U\TRAIN/255;
% XX2=U*Vt;
% xl=1:1:6;str=['训练样本数' ];
% figure('NumberTitle', 'off', 'Name', 'PNGE识别率');
% plot(xl,acc,'-*b');title('识别率随训练样本的变化');xlabel(str);ylabel('识别率');
% axis([1,6,0,1]);
% legend('PNGE','Location','southeast');
% set(gca, 'yTick', 0:0.1:1);
%%  画图
figure;
plot([1 : 1000], J);
title('Convergence function of CMU PIE dataset');
ylabel('Objective function values');
xlabel('Number of iterations');
set(gcf,'unit','centimeters','position',[3 5 16 12]);
% saveas(gcf,'E:\desktop\7.29汇报\LRPFEPRNF2收敛.jpg');
Ffig2('E:\desktop\第一篇\2.0图\4.7\CMUPIE收敛.png',25,10);
%%
function Ffig2(filename,borderSize,b2)
    % 获取图形的内容部分（去除边框）
    frame = getframe(gcf);
    % 裁剪边框
%     figure;
    frameDataCropped = frame.cdata(b2:end-b2-1, borderSize:end-borderSize-5, :);
    frameDataCropped(frameDataCropped==240)=255;
%     imshow(frameDataCropped);
    imwrite(frameDataCropped, filename);
%     saveas(gcf,filename);
end
