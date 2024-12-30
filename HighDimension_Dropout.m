%-------------------------------------------------------------------------
% This is the matlab implementation of the Dropout approach according to
% the following work. The Dropout approach randomly select a subset
% variables to optimize in each BO iteration. Instead of training the GP
% model in the subspace, I train the GP in the original space in the
% implemetation since I found training GP in the original high-dimensional
% space can yield better results.
% Reference:
% C. Li, S. Gupta, S. Rana, T. V. Nguyen, S. Venkatesh, and A. Shilton,
% High dimensional Bayesian optimization using dropout. International Joint
% Conference on Artifical Intelligence, 2017, 2096-2102.
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
fprintf('Dropout on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation <  max_evaluation
    optimized_index = randperm(num_vari,sub_vari);
    % build the GP model
    train_x = sample_x(:,optimized_index);
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % find the point with the highest EI value using GA algorithm
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
    fprintf('Dropout on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end



