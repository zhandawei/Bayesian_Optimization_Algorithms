%-------------------------------------------------------------------------
% This is the matlab implementation of the expected coordinate improvement
% approach according to the following work.
% Reference:
% D. Zhan. Expected coordinate improvement for high-dimensional Bayesian
% optimization. Swarm and Evolutionary Computation. 2024, 91: 101745.
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
fprintf('ECI on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation <  max_evaluation
    % train the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % find the maximum ECI values for all the coordinates
    max_ECI = zeros(1,num_vari);
    max_x = zeros(1,num_vari);
    % you can use parfor to compute this in parallel
    for ii = 1:num_vari
        [max_x(ii),max_ECI(ii)] = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,xmin,ii),1,lower_bound(ii),upper_bound(ii),10,20);
    end
    % get the coordinate optimization order based on the ECI values
    [sort_EI,sort_dim] = sort(-max_ECI,'descend');
    % optimize one coordinate at a time
    for ii = 1:num_vari
        optimized_index = sort_dim(ii);
        if ii == 1
            optimized_x = max_x(optimized_index);
        else
            GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
            [optimized_x,EI] = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,xmin,optimized_index),1,lower_bound(optimized_index),upper_bound(optimized_index),10,20);
        end
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
        fprintf('ECI on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
    end
end



