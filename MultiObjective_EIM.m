%-------------------------------------------------------------------------
% This is the matlab implementation of EIM algorithm according to
% the following work. For the improvement function in EIM, you can choose
% 'Euclidean', 'Maximin', or 'Hypervolume'.
% Reference:
% D. Zhan, Y. Cheng, and J. Liu. Expected Improvement Matrix-Based Infill
% Criteria for Expensive Multiobjective Optimization. IEEE Transactions on
% Evolutionary Computation. 2017, 21: 956-975.
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
% number of initial design points
num_initial = 100;
% maximum number of evaluations
max_evaluation = 200;
% the intial design points, points sampled all at once
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound) + lower_bound;
sample_y = feval(fun_name, sample_x, num_obj);
% scale the objectives to [0,1]
sample_y_scaled =(sample_y - min(sample_y))./(max(sample_y)-min(sample_y));
% initialize some parameters
iteration = 0;
evaluation = size(sample_x,1);
GP_obj = cell(1,num_obj);
% calculate the initial hypervolume values
index = Paretoset(sample_y);
non_dominated_front = sample_y(index,:);
non_dominated_front_scaled = sample_y_scaled(index,:);
hypervolume = Hypervolume(non_dominated_front,ref_point);
% print the hypervolume information
fprintf('EIM on %d-D %s function, iteration: %d, evaluation: %d, hypervolume: %0.2f \n',num_vari,fun_name,iteration,evaluation,hypervolume);
% beginning of the iteration
while evaluation < max_evaluation
    % build the kriging model for each objective
    for ii = 1:num_obj
        GP_obj{ii} = GP_Train(sample_x,sample_y_scaled(:,ii),lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    end
    % select updating points using the EIM criteria
    % you can choose 'Euclidean', 'Maximin', and 'Hypervolume'
    criterion = 'Euclidean';
    infill_x = Optimizer_GA(@(x)-Infill_EIM(x,GP_obj,non_dominated_front_scaled,'Euclidean'),num_vari,lower_bound,upper_bound,10*num_vari,200);
    infill_y = feval(fun_name,infill_x,num_obj);
    % add the new points to the design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    sample_y_scaled = (sample_y - min(sample_y))./(max(sample_y)-min(sample_y));
    evaluation = evaluation + size(infill_x,1);
    iteration = iteration + 1;
    % calculate the hypervolume values
    index = Paretoset(sample_y);
    non_dominated_front = sample_y(index,:);
    non_dominated_front_scaled = sample_y_scaled(index,:);
    hypervolume = Hypervolume(non_dominated_front,ref_point);
    % print the hypervolume information
    fprintf('EIM on %d-D %s function, iteration: %d, evaluation: %d, hypervolume: %0.2f\n',num_vari,fun_name,iteration,evaluation,hypervolume);
end

