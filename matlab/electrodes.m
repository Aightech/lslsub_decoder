function [] = electrodes(fig,data,t)
    figure(fig);
    cla reset;   
    
    subplot(3,3,1);
    cla reset;
    %surf(reshape(data(t,1:64),[8 8]))
    %hold on
    s=pcolor(reshape(data(t,1:64),[8 8]));
    s.FaceColor = 'interp';
    caxis([0 0.2]);
    
    
    subplot(3,3,2);
    cla reset;
    s=pcolor(reshape(data(t,1+64*1:64*2),[8 8]));
    s.FaceColor = 'interp';
    caxis([0 0.2]);
    
    
    subplot(3,3,3);
    cla reset;
    s=pcolor(reshape(data(t,1+64*2:64*3),[8 8]));
    s.FaceColor = 'interp';
    caxis([0 0.2]);
    
    
    subplot(3,3,[4 5 6]);
    cla reset;
    s=pcolor(reshape([0 data(t,1+64*3:64*4)],[5 13]));
    s.FaceColor = 'interp';
    caxis([0 0.2]);
    
    subplot(3,3,[7 8 9]);
    cla reset;
    plot(data(:, 1:end-1));
    hold on 
    plot(data(:,end)./data(:,end)+2);
    hold on 
    plot([t t], ylim);
end

