clear all;
close all;
name={};
name=[name,'result'];
a=load('result.mat');
predicts=a.metrics(1,:);
predicts=reshape(cell2mat(predicts),[29,15860])';
min(min(predicts))
predicts=(predicts-0.5)/100;
labels=cell2mat(a.metrics(2,:))/100;
mcs=cell2mat(a.metrics(3,:));

l=size(predicts,1);

close all
found=false;
maxi=max(predicts,[],2);
mini=min(predicts,[],2);
diff=maxi-mini;
[b,inx]=sort(diff,'descend');


for i=41:50
k=inx(i);

k=randi(length(labels));
while predicts(k,end)>predicts(k,1)+1 
    k=randi(length(labels));
end
k
figure('color',[1 1 1]);
plot(predicts(k,:),'k-<')
hold on
plot(mcs(k),labels(k),'h','MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',12)
ll=legend('prediction','label');
xlabel('MCS')
ylabel('Data rate (Mb/s)')
grid on
doit
end
