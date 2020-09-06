clear all;
close all;


a=load('result_ack_alg.mat');


pr=a.metrics(1,:);
lp=length(pr);
pr=reshape(cell2mat(pr),[29,lp])';

ef=a.metrics(2,:);
ef=reshape(cell2mat(ef),[29,lp])';

mcs=cell2mat(a.metrics(4,:));


[b,inx]=sort(ef(:,end),'descend');

for i=1:10
    k=inx(i);
    k=randi(lp);
    
    found=false;
%     while found==false
%         k=randi(lp);
%         [maxi,maxii]=max(ef(k,:));
%         if maxii<5
%             found=true;
%         end
%         
%     end
    
    found=false;
    figure;
    plot(ef(k,:)/1000,'k-o')
    hold on;
    ydata=get(gca,'YLim');
    plot(mcs(k)*ones(1,10),linspace(0,ydata(2),10),'r--')
    
    ll=legend('ACK\_NET','Simulator-MCS');
    xlabel('MCS')
    ylabel('Data rate (Mb/s)')
    grid on
    doit
    
    
end





