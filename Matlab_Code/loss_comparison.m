clear all;
close all;
name={};
name=[name,'result_njf_mse_tl_alg'];
name=[name,'result_njf_mae_tl_alg'];
name=[name,'result_njf_mape_tl_alg'];
pro={'MSE_','MAE_','MAPE_','SAME_'};
for j=1:length(name)
    
    tname=[name{j},'.mat'];
    tempdata=load(tname);
    eval([pro{j},'predicts=tempdata.predicts;']);
    eval([pro{j},'labels=tempdata.labels;']);
    
end

[l,w]=size(MSE_labels);
k=randperm(w);


offset=0.5;
for j=1:length(name)
    eval([pro{j},'predicts=','max(',pro{j},'predicts(k)-offset,0);']);
    eval([pro{j},'labels=',pro{j},'labels(k)-offset;']);
end

for j=1:length(name)
    eval([pro{j},'predicts=',pro{j},'predicts(:);']);
    eval([pro{j},'labels=',pro{j},'labels(:);']);
end

label=MAPE_labels;


mean(label)





h(1,1)=cdfplot(abs(MSE_predicts-label)./(MSE_predicts+label+1)*100);
hold on;
h(1,2)=cdfplot(abs(MAE_predicts-label)./(MAE_predicts+label+1)*100);
h(1,3)=cdfplot(abs(MAPE_predicts-label)./(MAPE_predicts+label+1)*100);
set( h(1,:), 'LineWidth',1);
xlabel('Error Percentage (%)')
ylabel('F(x)')
title('CDF')
ll=legend('MSE','MAE','MAPE');


doit
