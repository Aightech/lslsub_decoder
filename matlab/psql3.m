
subplot(2,3,1);
s=pcolor(reshape(trainedModel1.PCACenters(t,1:64),[8 8]));
s.FaceColor = 'interp';
caxis([0 0.3]);


subplot(2,3,2);
s=pcolor(reshape(trainedModel1.PCACenters(t,1+64*1:64*2),[8 8]));
s.FaceColor = 'interp';
caxis([0 0.3]);


subplot(2,3,3);
s=pcolor(reshape(trainedModel1.PCACenters(t,1+64*2:64*3),[8 8]));
s.FaceColor = 'interp';
caxis([0 0.3]);


subplot(2,3,[4 5 6]);
cla reset;
s=pcolor(reshape([0 trainedModel1.PCACenters(t,1+64*3:64*4)],[5 13]));
caxis([0 0.3]);