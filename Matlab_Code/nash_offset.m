clear all
close all
load('exp_data.mat')


c1=UAT(1:2:size(UAT,1),:);
c2=UAT(2:2:size(UAT,1),:);
cc=c2>=c1;
sum(cc,1)


d1=THRUPUT(1:2:length(THRUPUT));
d2=THRUPUT(2:2:length(THRUPUT));
sum(d1>d2)

a=OLLA;
b=a;
b(:,1)=[];
b=b(:);
histogram(b,20);
figure;
histogram(a(2:2:size(a,1),1),20);
figure;

cdfplot(b)