%-------------------------------------------------------------------------
% This is the matlab implementation of MOEA/D-EGO algorithm according to
% the following work.
% Reference:
% Q. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive Multiobjective
% Optimization by MOEA/D with Gaussian Process model. IEEE Transactions on
% Evolutionary Computation. 2010, 14, 456-474.
% Author: Dawei Zhan
% Date:   2024/12/30
%-------------------------------------------------------------------------
clearvars;close all;
% settings of the problem
fun_name = 'DTLZ2';
% number of objectives
num_obj = 2;
% number of design variables
num_vari = 10;
lower_bound = zeros(1,num_vari);
upper_bound = ones(1,num_vari);
% reference point for calculating hypervolume
ref_point = 2.5*ones(1,num_obj);
% number of points evaluated in each iteration
Ke = 5;
% number of initial designs
num_initial = 100;
% maximum number of evaluations
max_evaluation = 200;
% initial sampling
sample_x = lower_bound + (upper_bound-lower_bound).*lhsdesign(num_initial, num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name,sample_x,num_obj);
z = min(sample_y);
% the number of weight vectors
if num_obj == 2
    n = 300;
elseif num_obj == 3
    n = 595;
else
    n = 800;
end
% generate weights
weight = UniformPoint(n,num_obj);
n = size(weight,1);
iteration = 0;
evaluation = size(sample_x,1);
GP_obj = cell(1,num_obj);
index = Paretoset(sample_y);
non_dominated_front = sample_y(index,:);
hypervolume = Hypervolume(non_dominated_front,ref_point);
fprintf('MOEA/D-EGO on %d-D %s function, iteration: %d, evaluation: %d, hypervolume: %0.2f\n',num_vari,fun_name,iteration,evaluation,hypervolume);
while evaluation < max_evaluation
    q = min(max_evaluation-evaluation,Ke);
    for ii = 1:num_obj
        GP_obj{ii} = GP_Train(sample_x,sample_y(:,ii),lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    end
    % optimize n EIs using MOEA/D
    T = 20;
    B       = pdist2(weight,weight);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);
    pop_vari = rand(n,num_vari).*(upper_bound-lower_bound) + lower_bound;
    pop_EI = zeros(n,1);
    pop_y = zeros(n,num_obj);
    pop_s = zeros(n,num_obj);
    for ii = 1:n
        for jj = 1:num_obj
            [pop_y(ii,jj),pop_s(ii,jj)] = GP_Predict(pop_vari(ii,:),GP_obj{jj});
        end
        pop_EI(ii) =  Tchebycheff_EI(pop_y(ii,:),pop_s(ii,:),weight(ii,:),z,sample_y);
    end
    for gen = 2:round(10000/n)
        for ii = 1:n
            parent = pop_vari(ii,:);
            if rand < 0.9
                P = B(ii,randperm(T));
            else
                P = randperm(n);
            end
            F = 0.5;
            CR = 0.8;
            mutation = parent + F*(pop_vari(P(1),:)-pop_vari(P(2),:));
            if any(mutation <lower_bound) || any(mutation>upper_bound)
                mutation = lower_bound + rand(1,num_vari).*(upper_bound-lower_bound);
            end
            rand_vector = rand(1,num_vari);
            rand_vector(1,randi(num_vari)) = 0;
            mui = rand_vector < CR;
            offspring = mutation.*mui + parent.*(1-mui);

            offspring_y = zeros(1,num_obj);
            offspring_s = zeros(1,num_obj);
            for jj = 1:num_obj
                [offspring_y(1,jj),offspring_s(1,jj)] = GP_Predict(offspring,GP_obj{jj});
            end
            offspring_EI = Tchebycheff_EI(offspring_y,offspring_s,weight(P,:),z,sample_y);
            replace_index = P(offspring_EI > pop_EI(P,:));
            c = min(2,length(replace_index));
            replace_index = replace_index(1:c);
            if ~isempty(replace_index)
                pop_vari(replace_index,:) = repmat(offspring,c,1);
                pop_y(replace_index,:) = repmat(offspring_y,c,1);
                pop_s(replace_index,:) = repmat(offspring_s,c,1);
                pop_EI(replace_index) = offspring_EI(1:c);
            end
        end
    end
    % delete points that are too close to each other
    temp_point = sample_x;
    candi_index = [];
    for ii = 1:n
        if min(pdist2(pop_vari(ii,:),temp_point)) > 1E-5
            candi_index = [candi_index;ii];
        end
    end
    candi_point = pop_vari(candi_index,:);
    candi_weight = weight(candi_index,:);
    candi_y = pop_y(candi_index,:);
    candi_s = pop_s(candi_index,:);
    % clustering
    idx = kmeans(candi_weight,q);
    infill_x = zeros(q,num_vari);
    for ii = 1:q
        group_point = candi_point(idx==ii,:);
        group_weight = candi_weight(idx==ii,:);
        group_y = candi_y(idx==ii,:);
        group_s = candi_s(idx==ii,:);
        EI = zeros(size(group_point,1),1);
        for jj = 1:size(group_point,1)
            EI(jj) = Tchebycheff_EI(group_y(jj,:),group_s(jj,:),group_weight(jj,:),z,sample_y);
        end
        [~,select_index] = max(EI);
        infill_x(ii,:) = group_point(select_index,:);
    end
    infill_y = feval(fun_name,infill_x,num_obj);
    evaluation = evaluation + size(infill_y,1);
    iteration = iteration + 1;
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update z
    z = min(sample_y);
    index = Paretoset(sample_y);
    non_dominated_front = sample_y(index,:);
    hypervolume = Hypervolume(non_dominated_front,ref_point);
    fprintf('MOEA/D-EGO on %d-D %s function, iteration: %d, evaluation: %d, hypervolume: %0.2f\n',num_vari,fun_name,iteration,evaluation,hypervolume);
end










