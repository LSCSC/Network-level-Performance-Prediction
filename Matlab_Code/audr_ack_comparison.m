clear all;
close all;

b=load('result.mat');
predicts=b.metrics(1,:);
predicts=reshape(cell2mat(predicts),[29,15860])';
min(min(predicts))
predicts=(predicts-0.5)/100;
labels=cell2mat(b.metrics(2,:))/100;
mcs=cell2mat(b.metrics(3,:));

a=load('result_ack.mat');
pr=a.metrics(1,:);
pr=reshape(cell2mat(pr),[29,15860])';

ef=a.metrics(2,:);
ef=reshape(cell2mat(ef),[29,15860])';

l=size(predicts,1);

close all
found=false;
maxi=max(predicts,[],2);
mini=min(predicts,[],2);
diff=maxi-mini;
[b,inx]=sort(diff,'descend');


for i=20:30
k=inx(i);
k=randi(15860);
figure('color',[1 1 1]);
m=max(predicts(k,:));
plot(predicts(k,:)/m,'r-<')
hold on
%plot(mcs(k),labels(k)/m,'h','MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',12)

m=max(ef(k,:));
plot(ef(k,:)/m,'k-o')

ll=legend('AUDR\_NET','ACK\_NET');
xlabel('MCS')
ylabel('Normalized AUDR')
grid on
doit
end





