clear all;
close all;

b=load('hist_njf_tl_ack_random.mat');


b=b.metrics;



b1=b(1,:);
b2=b(2,:);
b3=b(3,:);
b4=b(4,:);


plot(b1,'r--')
hold on
plot(b3,'k')
ll=legend('VAL loss','Training loss');
xlabel('Epoch index')
title('Loss')
grid on
doit

figure;
plot(b2,'r--')
hold on
plot(b4,'k')
ll=legend('VAL acc','Training acc');
xlabel('Epoch index')
title('Accuracy')
grid on
doit