close all
clear all
a=load('hist_training_loss.mat');
model_a=a.metrics(1,:);
model_b=a.metrics(2,:);

plot(model_a,'k-')
hold on
plot(model_b,'r--')
ylabel('Training Loss - MAPE (%)')
xlabel('Epoch index')
legend('AUDR\_NET\_A','AUDR\_NET\_B')

grid on
doit