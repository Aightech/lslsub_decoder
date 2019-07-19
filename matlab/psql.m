%clear()
%% IMPORT DATA
session_index=5;
parameters.id=[12 14 17 19 21];%session 
parameters.start_mvt=[6.5 -0.50 6 14.5 11.7];
parameters.size_mvt=[6 6 6 6 6];
parameters.win_size=[2.7 2.5 2.5 2.5 2.5];
parameters.start_mvt2=[4.5 -2.7 4 12.5 9.7];
parameters.size_mvt2=[6 6 6 6 6];
parameters.win_size2=[0.8 0.8 0.8 0.8 0.8];


otb = importdata(strcat('tne_project/database/lslsub_dbfeeder/otb408_id',string(parameters.id(session_index)),'.csv'));
glv = importdata(strcat('tne_project/database/lslsub_dbfeeder/glv_id',string(parameters.id(session_index)),'.csv'));
 

%% PREPARE DATA
t_start = 0;
t_end = t_start + 8*19;


array_otb = sortrows(otb,1);
times_otb = array_otb(:,2);
sync_otb = array_otb(:,end-6)/32000;
count_otb = array_otb(:,end-7);
data_otb = array_otb(:,2+8*16+1:end-8-16);

t_offset = times_otb(1);
times_otb = times_otb - t_offset;
index_t_otb = [1,1];
while(times_otb(index_t_otb(1)) < t_start && index_t_otb(1) < size(times_otb,1))
    index_t_otb(1)=index_t_otb(1)+1;
end
index_t_otb(2) = index_t_otb(1);
while(times_otb(index_t_otb(2)) < t_end && index_t_otb(2) < size(times_otb,1))
    index_t_otb(2)=index_t_otb(2)+1;
end

array_glv = sortrows(glv,1);
times_glv = array_glv(:,2);
sync_glv = array_glv(:,end);
data_glv = array_glv(:,3:end-1);

times_glv = times_glv - t_offset;
index_t_glv = [1,1];
while(times_glv(index_t_glv(1)) < t_start && index_t_glv(1) < size(times_glv,1))
    index_t_glv(1)=index_t_glv(1)+1;
end
index_t_glv(2) = index_t_glv(1);
while(times_glv(index_t_glv(2)) < t_end && index_t_glv(2) < size(times_glv,1))
    index_t_glv(2)=index_t_glv(2)+1;
end

labels_otb = zeros(size(times_otb));
for i =0:18
    
    labels_otb(times_otb(:) > parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) & times_otb(:) < parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index)) = i+1;
end
data_train=[data_otb labels_otb];


%% PROCESS DATA

sat=35000;
outlayer=[];
power=zeros(size(data_otb,2),1);
electrode = zeros(8,3*8);
electrode2 = zeros(5,13);
for s=1:size(data_otb,2)
    for i=index_t_otb(1):index_t_otb(2)
        if(abs(data_otb(i,s))>sat)
            outlayer = [outlayer s];
            power(s)=NaN;
            break;
        else
            power(s)=power(s)+abs(data_otb(i,s)/sat)^2;    
        end
    end
    power(s)=log(power(s)/(index_t_otb(2)-index_t_otb(1)));
end

for s=1:size(data_otb,2)
    if(s<=3*64)
        electrode(s)=power(s);
    else
        electrode2(s-3*64+1)=power(s);
    end
end

figure(1)
subplot(2,3,1);
heatmap(electrode(:,1+16:8*3));
subplot(2,3,2);
heatmap(electrode(:,1+8:8*2));
subplot(2,3,3);
heatmap(electrode(:,1:8));
subplot(2,3,[4 5 6]);
heatmap(electrode2);




for s=1:size(outlayer,2)
    for i=index_t_otb(1):index_t_otb(2)
        data_otb(i,outlayer(s))=NaN;
    end
end

%%
figure(2)
clf
for i=1:4
    subplot(5,1,i);
    plot(times_otb(index_t_otb(1):index_t_otb(2)), data_otb(index_t_otb(1):index_t_otb(2),1+64*(i-1):64*i));
    ylim([-5000 5000]);
    lim = ylim;
    hold on
    for i = 0:18
        harea = area([parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index)], ones(1,2)*lim(1) ,lim(2), 'LineStyle', 'none');
        set(harea, 'FaceColor', 'r');
        alpha(0.25);
        hold on
    end
end
subplot(5,1,5)
plot(times_glv(index_t_glv(1):index_t_glv(2)), data_glv(index_t_glv(1):index_t_glv(2),:))
lim = ylim;
hold on
for i =0:18
    harea = area([parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index)], ones(1,2)*lim(1) ,lim(2), 'LineStyle', 'none');
    set(harea, 'FaceColor', 'r');
    alpha(0.25);
    hold on
end

%%
figure(3)
plot(times_otb(index_t_otb(1):index_t_otb(2)), sync_otb(index_t_otb(1):index_t_otb(2)));
hold on
plot(times_glv(index_t_glv(1):index_t_glv(2)), sync_glv(index_t_glv(1):index_t_glv(2)));

%%
figure(4)
clf
subplot(2,1,1);
plot(times_glv(index_t_glv(1):index_t_glv(2)), data_glv(index_t_glv(1):index_t_glv(2),:))
hold on
lim = ylim; 
for i =0:18
    harea = area([parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index)], ones(1,2)*lim(1) ,lim(2), 'LineStyle', 'none');
    set(harea, 'FaceColor', 'r');
    alpha(0.25);
    hold on
end
plot(times_otb(index_t_otb(1):index_t_otb(2)), labels_otb(index_t_otb(1):index_t_otb(2),:)*10);

subplot(2,1,2);
plot(times_glv(index_t_glv(1):index_t_glv(2)), data_glv(index_t_glv(1):index_t_glv(2),1:3));
legend();

%%

windowSize = 1000; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
clf

ch_min=1;
ch_max=256;
sample_average_nb=100;
mean_otb=mean(data_otb(index_t_otb(1):index_t_otb(2),ch_min:ch_max));
max_otb = max(data_otb(index_t_otb(1):index_t_otb(2),ch_min:ch_max)-mean_otb);
process_otb = filter(b, a, abs(data_otb(index_t_otb(1):index_t_otb(2),ch_min:ch_max)-mean_otb)./max_otb);

kept_otb=[];
training_point=[];
   

for i =0:18
    data=process_otb(times_otb(:)>parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) & times_otb(:)<parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index), ch_min:ch_max);
    kept_otb = [kept_otb ; data ones(size(data,1),1)*(i+1)];
    j=0;
    while j+sample_average_nb<size(data,1)
        training_point = [training_point ; mean(data(j+1:j+sample_average_nb-1,:))  (i+1) ];
        j=j+sample_average_nb;
    end
    %process_otb(times_otb(:)>parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) & times_otb(:)<parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index), ch_min:ch_max) = 1;
    %process_otb(times_otb(:)>parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) & times_otb(:)<parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index), ch_min:ch_max) =process_otb(times_otb(:)>parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) & times_otb(:)<parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index), ch_min:ch_max).*mean(data);
    data=process_otb(times_otb(:)>parameters.start_mvt2(session_index)+i*parameters.size_mvt2(session_index) & times_otb(:)<parameters.start_mvt2(session_index)+parameters.win_size2(session_index)+i*parameters.size_mvt2(session_index), ch_min:ch_max);
    kept_otb = [kept_otb ; data zeros(size(data,1),1)];
    j=0;
    while j+sample_average_nb<size(data,1)
        training_point = [training_point ; mean(data(j+1:j+sample_average_nb-1,:)) 0 ];
        j=j+sample_average_nb;
    end
end


plot(times_otb(index_t_otb(1):index_t_otb(2)),process_otb);
hold on
plot(times_glv(index_t_glv(1):index_t_glv(2)), data_glv(index_t_glv(1):index_t_glv(2),:)/650+0.5);

hold on
lim = ylim; 
for i =0:18
    harea = area([parameters.start_mvt(session_index)+i*parameters.size_mvt(session_index) parameters.start_mvt(session_index)+parameters.win_size(session_index)+i*parameters.size_mvt(session_index)], ones(1,2)*lim(1) ,lim(2), 'LineStyle', 'none');
    set(harea, 'FaceColor', 'r');
    alpha(0.05);
    hold on
end


for i =0:19
    harea = area([parameters.start_mvt2(session_index)+i*parameters.size_mvt2(session_index) parameters.start_mvt2(session_index)+parameters.win_size2(session_index)+i*parameters.size_mvt2(session_index)], ones(1,2)*lim(1) ,lim(2), 'LineStyle', 'none');
    set(harea, 'FaceColor', 'b');
    alpha(0.05);
    hold on
end

csvwrite('training_point_id'+string(parameters.id(session_index))+'.csv',training_point)

