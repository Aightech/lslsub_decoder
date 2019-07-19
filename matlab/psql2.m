ids=[12 14 17 19 21];%

nb_sess=4
training_point = [];
for i =1:nb_sess
    train = importdata('training_point_id'+string(ids(i))+'.csv');
    training_point = [ training_point; train]; 
end
training_point_y = training_point(:,end);
training_point_x = training_point(:,1:end-1);

test_point=[]
for i =nb_sess+1:5
    i
    train = importdata('training_point_id'+string(ids(i))+'.csv');
    test_point = [ test_point; train]; 
end
test_point_y = test_point(:,end);
test_point_x = test_point(:,1:end-1);


%MdlLinear = fitcdiscr(training_point_x,training_point_y);
% 
% rng(1)
% MdlLinear = fitcdiscr(training_point_x,training_point_y,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%      struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false))
f = figure;
ax = axes('Parent',f,'position',[0.13 0.39  0.77 0.54]);



t=1;

electrodes(f,test_point,t);

b = uicontrol('Parent',f,'Style','slider','Position',[95,5,429,23],...
              'Value',t, 'min',0, 'max',size(test_point,1));
      
addlistener(b,'Value','PostSet', @(~,b) electrodes(f,test_point,round(b.AffectedObject.Value))); 


figure(2)
plot(predict(MdlLinear,[training_point_x(:,:); test_point_x(:,:)])+0.5);
hold on; 
plot([training_point_y(:,:); test_point_y(:,:)]);
