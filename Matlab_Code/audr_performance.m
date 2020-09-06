clear all;
close all;
name=[];
name=[name,'result_njf_cl_coteach_mape_allmetric_alg_789'];
% for j=1:length(name)
% for i=0:4
%     tname=[name,'_',num2str(i),'.mat']
%     disp(['a_',num2str(i),'=load(',tname,')'])
%     eval(['a_',num2str(i),'=load(',tname,')']);
% end
% end

a=load('result_njf_cl_coteach_mape_allmetric_alg_789.mat');

a=a.metrics;



[l,w]=size(a);



for ww=1:w
   predicts(ww)=a(1,ww);
end

for ww=1:w
   label(ww)=a(2,ww);
end
k=randperm(w);

predicts=max(predicts(k),0);
label=max(label(k),0);

while true
    disp('finding...')
    if max(label(540:599))>20 && max(label(540:599))<300 && sum(label(540:599)>80)>6
        break;
    end 
    k=randperm(w);
predicts=max(predicts(k),0);
label=max(label(k),0);
end


figure('color',[1 1 1]);
plot(predicts(540:599)/100,'r-')
hold on
plot(label(540:599)/100,':','Color',[0.3 0.3 0.3],'Linewidth',1)
ll=legend('prediction','label');

xlabel('Epoch index')
ylabel('Data rate (Mb/s)')
legend('AUDR\_NET','Simulator')
grid on 
doit

figure;

h(1,1)=cdfplot(predicts/100);
hold on;
h(1,2)=cdfplot(label/100);
set(gca,'XScale','log')
xlim([0.02,15])
xlabel('UAT (Mb/s)')
ylabel('F(x)')
title('CDF')
ll=legend('UATNet','Simulator');
max(predicts/100)
max(label/100)
% figure('color',[1 1 1]);
% h(1,1)=cdfplot(abs(MSE_predicts-label)./(MSE_predicts+label+1)*100);
% hold on;
% h(1,2)=cdfplot(abs(MAE_predicts-label)./(MAE_predicts+label+1)*100);
% h(1,3)=cdfplot(abs(MAPE_predicts-label)./(MAPE_predicts+label+1)*100);
% set( h(1,:), 'LineWidth',1);
% set(ll,'FontSize',15);
% xlabel('Error Percentage (%)','FontSize',15)
% ylabel('F(x)','FontSize',15)
% title('Error Percentage CDF','FontSize',15)
% ll=legend('MSE','MAE','MAPE');
% set(ll,'FontSize',15);


% figure
% plot(smooth(abs(ma7(:,end))));
% hold on;
% plot(smooth(abs(mb7(:,end))));
% plot(smooth(abs(mc7(:,end))));
% legend('cl+coteach+mape','cl+mape','mape only')
% xlim([0 80])
% ylim([0.12 0.25])

% mean(label)
% mean(predicts)
% figure('color',[1 1 1]);
% subplot(1,2,1)
% plot(predicts(540:599)/100,'r-')
% hold on
% plot(label(540:599)/100,'b--')
% ll=legend('prediction','label');
% set(ll,'FontSize',15);
% xlabel('Epoch index','FontSize',15)
% ylabel('Data rate (Mb/s)','FontSize',15)
% title('Error Percentage','FontSize',15)
% 
% 
% subplot(1,2,2)
% cdfplot(abs(label)/100)
% % ll=legend('cl+coteach+mape','cl+mape','mape only');
% set(ll,'FontSize',15);
% xlabel('Label Data Rate (Mb/s)','FontSize',15)
% title('Label CDF','FontSize',15)
% 
% figure('color',[1 1 1]);
% subplot(1,2,1)
% cdfplot(abs(max(predicts-1,0)-label+1)./(max(predicts-1,0)+label-1+1)*100)
% % ll=legend('cl+coteach+mape','cl+mape','mape only');
% set(ll,'FontSize',15);
% xlabel('Error Percentage (%)','FontSize',15)
% title('Error Percentage CDF','FontSize',15)
% 
% 
% subplot(1,2,2)
% cdfplot(abs(predicts-label)/100)
% % ll=legend('cl+coteach+mape','cl+mape','mape only');
% set(ll,'FontSize',15);
% xlabel('Error (Mb/s)','FontSize',15)
% xlim([0 2])
% title('Absolute Error CDF','FontSize',15)
% 
% disp(min(label))
% disp(min(predicts))
% 
% mean(abs(max(predicts-1,0)-label+1)./(max(predicts-1,0)+label-1+2)*100)
% 
% 
% mean(abs(MAPE_predicts-label)./(MAPE_predicts+label+1)*100)
