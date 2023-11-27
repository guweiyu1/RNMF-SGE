function [U,W,Y,Vt,class_recT,rc] = RNMF_SGE_mse( TRAIN, class_Train, m_img, n_img, Train_num, r, maxiter,sigma, TEST, class_Test, Test_num, classnum, totlenuminclass)
%V-----------------ѵ�����ݼ���[m*n, Train_num*15]
%class_V-----------ѵ��������Ӧ�ķ���
%m_img-------------V��ͼ���ά��
%n_img-------------V��������
%r-----------------V����
%Train_num---------ѵ��������
%maxiter-----------����������
%T-----------------�������ݼ���[m*n, Test_num*15]
%class_T-----------����������Ӧ�ķ���
%Test_num----------ѵ��������

%% ѵ��
J = zeros(maxiter, 1);
TRAIN = max(TRAIN,1e-9);
TEST = max(TEST,1e-9);
U = abs(randn(m_img * n_img, r));                                          %�Ǹ���ʼ��
Y = abs(randn(Train_num * classnum, r));
%% W
A1 = ones(Train_num,Train_num);W = ones(Train_num,Train_num);
for i=2:classnum
    W = blkdiag(W,A1);
end
% for i=1:Train_num * classnum
%     W(i,i)=0;
% end
for i=1:Train_num * classnum
    for j=1:Train_num * classnum
        if W(i,j)==1
            W(i,j) = exp(-norm(TRAIN(:,i)-TRAIN(:,j))^2 / (2 * sigma ^ 2));%������
%             if W(i,j)>1e-4
%                 fprintf('%f\n', exp(-norm(TRAIN(:,i)-TRAIN(:,j))^2 / (2 * sigma ^ 2)));
%             end
        end
    end
end
% W = ConstructAdjacencyGraphinclassgai(TRAIN,totlenuminclass,Train_num, 20,0,sigma);%�������ǩ��ò�Ҫ��K����
W = sparse(W);              %ϡ�軯 �Ż��㷨���Ӷ�
%%
% norms = sqrt(sum((W*Y).^2));                    %��ʼ��һ��
% Y = Y./(ones(Train_num * classnum,1) * norms);
% U = U.*(ones(m_img * n_img, 1)*norms);
% norms = max(1e-15,sqrt(sum(U.^2,1)))';
% U = U*spdiags(norms.^-1,0,r,r);
% Y = Y*spdiags(norms,0,r,r);
%% D
D=zeros(m_img * n_img);
    DD = TRAIN;
    for i=1:m_img * n_img
        D(i,i)=1 / norm(DD(i,:));
    end
D = sparse(D);                %ϡ�軯 �Ż��㷨���Ӷ�
J(1) = 0.5 * sum(sum((DD).^2));                                     %���ۺ���Ϊŷ�Ͼ���
%%
for iter = 1: maxiter
    DXWY = D * ( TRAIN * (W*Y) );
    DUYWWY = D * ( U * (Y'*(W'*W)*Y) );
    U = U.* DXWY./max(DUYWWY,1e-9);
    WXDU = W' * ( TRAIN' * (D*U) );
    WWYUDU = W' * W * Y * (U'*D*U) ;
    Y = Y.* WXDU./max(WWYUDU,1e-9);

%     norms = max(1e-15,sqrt(sum(U.^2,1)))';
%     U = U*spdiags(norms.^-1,0,r,r);
%     Y = Y*spdiags(norms,0,r,r);
    
%     norms = sqrt(sum((W*Y).^2));                    %��һ��
%     Y = Y./(ones(Train_num * classnum,1) * norms);
%     U = U.*(ones(m_img * n_img, 1)*norms);
    %% D
    %DD=(TRAIN-U*(W*Y)')*(TRAIN-U*(W*Y)')';
    DD = TRAIN-U*(W*Y)';
    for i=1:m_img * n_img
        D(i,i)=1 / norm(DD(i,:));
    end
    J(iter) = 0.5 * sum(sum(( DD).^2));                             %���´��ۺ���
    fprintf('%d\n', iter);
end
%     norms = sqrt(sum((W*Y).^2));                    %��һ��
%     Y = Y./(ones(Train_num * classnum,1) * norms);
%     U = U.*(ones(m_img * n_img, 1)*norms);

% norms = max(1e-15,sqrt(sum(U.^2,1)))';
% U = U*spdiags(norms.^-1,0,r,r);
% Y = Y*spdiags(norms,0,r,r);

% ������ۺ���������
figure;
plot([1 : maxiter], J);
title('Convergence function of Extended YALE-B database');
ylabel('Objective function values');
xlabel('Number of iterations');
set(gcf,'unit','centimeters','position',[3 5 16 12]);
saveas(gcf,'E:\desktop\��һƪ\2.0ͼ\4.7\YALEB����.jpg');

% figure;
% for i = 1 : r
%     subplot(8, 16, i);
%     im = reshape(W(:, i), m_img, n_img); 
%     imagesc(im);colormap('gray');  
% end

%% ����
%���������������ݱ�ʾΪW��ʸ�����������
% Vt = abs(randn(r, Test_num * classnum));
% norms = sqrt(sum(Vt'.^2));                              %��ʼ��һ��
% Vt = Vt./(norms'*ones(1,Test_num * classnum));
Vt = pinv(U) * TEST;

% Dt=zeros(m_img * n_img);
% DDt=TEST*TEST';
% for i=1:m_img * n_img
%     Dt(i,i)=(DDt(i,i))^0.5;
% end
% %%
% for iter = 1: maxiter
%     UDX = (U'*Dt*TEST);
%     UDUV = U'*Dt*U*Vt;
%     Vt = Vt.* UDX./UDUV;
% 
%     norms = sqrt(sum(Vt'.^2));              %��һ��
%     Vt = Vt./(norms'*ones(1,Test_num * classnum));
%     %% Dt
%     DDt=(TEST-U*Vt)*(TEST-U*Vt)';
%     for i=1:m_img * n_img
%         Dt(i,i)=(DDt(i,i))^0.5;
%     end
% end

%     norms = sqrt(sum(Vt'.^2));              %��һ��
%     Vt = Vt./(norms'*ones(1,Test_num * classnum));
% % rec_V = U * (W*Y)';
% % rec_T = U * Vt;                                                            %�ع�ͼ
%% ����ع�ͼ
% for i = 1 : Test_num * classnum
%     if mod(i, classnum) == 1
%         figure;
%         m = 1;
%     end
%     subplot(4, 5, m);
%     im = reshape(rec_T(:, i), m_img, n_img); 
%     imagesc(im);colormap('gray');  
%     m = m + 1;
% end
%% ����ƥ����
right = 0;
dist = zeros(1, Train_num * classnum);
class_recT = zeros(classnum * Test_num, 1);
WY = (W*Y)';
% WY = U \ TRAIN; 
for i = 1 : Test_num * classnum
    for j = 1 : Train_num * classnum
        dist(j) = norm(Vt(:, i) - WY(:, j));                              %ѡȡϵ����ŷ�Ͼ����������Ϊʶ�����
    end
    [mindist index] = sort(dist);
    class_recT(i) = class_Train(index(1));
end
for i = 1 : Test_num * classnum                                                  %ͳ��ʶ����
    if class_recT(i) == class_Test(i)
        right = right + 1;
    end
end
rc = right / (Test_num * classnum);
fprintf(2,'RNMF-SGE >>> r=%d , maxiter=%d , sigma=%.1f , Recognition Rate %.2f <<< RNMF-SGE\n',r,maxiter,sigma,100*rc);