%-------------------------------------------------------------------------
% This is the matlab implementation of Pareto Efficient Global Optimization
% algorithm according to the following work.
% Reference:
% J. Knowles. ParEGO: A Hybrid Algorithm With On-line Landscape Approximation
% for Expensive Multiobjective Optimization Problems. IEEE Transactions on
% Evolutionary Computation. 2006, 10: 50-66.
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
% the maximum allowed evaluations
max_evaluation = 200;
% generate weight vectors
if num_obj == 2
    num_weight = 11;
elseif num_obj == 3
    num_weight = 15;
elseif num_obj == 4
    num_weight = 20;
else
    num_weight = 56;
end
weight = UniformPoint(num_weight,num_obj);
% the intial design points, points sampled all at once
sample_x = lower_bound + (upper_bound-lower_bound).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name,sample_x,num_obj);
iteration = 0;
evaluation = size(sample_x,1);
% calculate the initial hypervolume value
index = Paretoset(sample_y);
non_dominated_front = sample_y(index,:);
hypervolume = Hypervolume(non_dominated_front,ref_point);
% print the hypervolume information
fprintf('ParEGO on %d-D %s problem, iteration: %d, evaluation: %d, hypervolume: %0.2f \n',num_vari,fun_name,iteration,evaluation,hypervolume);
% beginning of the iteration
while evaluation < max_evaluation
    % randomly select a weight vector
    lamda  = weight(randi(size(weight,1)),:);
    % build the weighted objective function
    sample_y_scaled = (sample_y-min(sample_y))./(max(sample_y)-min(sample_y));
    sample_y_pcheby = max(sample_y_scaled.*lamda,[],2) + 0.05*sum(sample_y_scaled.*lamda,2);
    % build the initial GP model
    GP_model = GP_Train(sample_x,sample_y_pcheby,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    %  GA is used for the maximization problem
    [infill_x,best_EI] = Optimizer_GA(@(x)-Infill_EI(x,GP_model,min(sample_y_pcheby)),num_vari,lower_bound,upper_bound,10*num_vari,200);
    % do the expensive evaluations
    infill_y = feval(fun_name,infill_x,num_obj);
    evaluation = evaluation + size(infill_y,1);
    iteration = iteration + 1;
    % add the evaluated points to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % plot current non-dominated front points
    index = Paretoset(sample_y);
    non_dominated_front = sample_y(index,:);
    hypervolume = Hypervolume(non_dominated_front,ref_point);
    fprintf('ParEGO on %d-D %s problem, iteration: %d, evaluation: %d, hypervolume: %0.2f \n',num_vari,fun_name,iteration, evaluation, hypervolume);
end