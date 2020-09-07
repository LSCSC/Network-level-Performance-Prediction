clear all;
close all;
name={};
name=[name,'result'];
a=load('result.mat');
predicts=(a.predicts-0.5)/100;
labels=a.labels/100;
mcs=a.mcs;
%
l=size(predicts,1);

close all
found=false;

for i=1:1
    while found==false
        k=randi(l);
        [maxi,maxii]=max(predicts(k,:));
        [mini,minii]=min(predicts(k,:));
        if maxii<20 && maxii>15 && mcs(k)>=13 && mcs(k)<=14 && predicts(k,1)<11 && predicts(k,end)>13 && predicts(k,end)<13.2
        %if maxi-mini>3
        %if mini<0.001 && maxi>0.8 && minii<=20
            found=true;
            disp(['found',num2str(i),'k',num2str(k)])
        end
    end
found=false;
figure
plot(predicts(k,:),'k-<')
hold on
plot(mcs(k),labels(k),'h','MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',12)
ll=legend('prediction','label');
xlabel('MCS')
ylabel('Data rate (Mb/s)')
grid on
doit
end