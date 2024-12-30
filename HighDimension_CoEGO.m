%-------------------------------------------------------------------------
% This is the matlab implementation of the cooperative efficient global
% optimization algorithm according to the following work.
% Reference:
% D. Zhan, J. Wu, H. Xing, and T. Li. A cooperative approach to efficient
% global optimization. Journal of Global Optimization. 2024, 88: 327-357.
% Author: Dawei Zhan
% Date:   2024/12/30
%-------------------------------------------------------------------------
clearvars; close all;
% setting of the problem
fun_name = 'Rosenbrock';
num_vari = 50;
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% number of variables to be optimized in each iteration
sub_vari = 5;
% the number of initial design points
num_initial = 200;
% maximum number of evaluations
max_evaluation = 500;
% initial design points using Latin hypercube sampling method
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound) + lower_bound;
sample_y = feval(fun_name,sample_x);
% the current iteration and evaluation
evaluation = size(sample_x,1);
iteration = 0;
% the current best solution
[fmin,index] = min(sample_y);
xmin = sample_x(index,:);
% print the current information to the screen
fprintf('CoEGO on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation <  max_evaluation
    % randomly divide the variables into num_vari/sub_vari groups
    rand_ind = randperm(num_vari);
    unsolved = min(max_evaluation-evaluation,num_vari/sub_vari);
    best_theta = ones(1,num_vari);
    for ii = 1:unsolved
        optimized_index = rand_ind((ii-1)*sub_vari+1:ii*sub_vari);
        % build the GP model cooperatively
        train_x = sample_x(:,optimized_index);
        GP_model = GP_Train_Cooperative(sample_x,sample_y,lower_bound,upper_bound,best_theta,0.001*ones(1,sub_vari),1000*ones(1,sub_vari),optimized_index);
        best_theta(optimized_index) = GP_model.theta(optimized_index);
        % optimize the EI cooperatively
        optimized_x = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,xmin,optimized_index),sub_vari,lower_bound(optimized_index),upper_bound(optimized_index),10*sub_vari,200);
        infill_x = xmin;
        infill_x(optimized_index) = optimized_x;
        % evaluate the query point with the real function
        infill_y = feval(fun_name,infill_x);
        % add the new point to design set
        sample_x = [sample_x;infill_x];
        sample_y = [sample_y;infill_y];
        % update some parameters
        evaluation = size(sample_x,1);
        iteration = iteration + 1;
        [fmin,index] = min(sample_y);
        xmin = sample_x(index,:);
        % print the current information to the screen
        fprintf('CoEGO on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
    end
end



