%--------------------------------------------------------------------------
% This is the pseudo expected improvement approach which uses an influence
% function to simulate the sequential EI's selection behavior. It is coded
% based on the following work.
% Reference: 
% D. Zhan, J. Qian, and Y. Cheng. Pseudo expected improvement
% criterion for parallel EGO algorithm. Journal of Global Optimization,
% 2017, 68(3): 641-662.
% Author: Dawei Zhan.
% Date:   2024.11.27
%--------------------------------------------------------------------------
clearvars; close all;
% setting of the problem
fun_name = 'Rosenbrock';
num_vari = 10;
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% the number of initial design points
num_initial = 20;
% maximum number of evaluations
max_evaluation = 120;
% the number of points selected in each iteration
num_q = 4;
% initial design points using Latin hypercube sampling method
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound) + lower_bound;
sample_y = feval(fun_name,sample_x);
% the current iteration and evaluation
evaluation = size(sample_x,1);
iteration = 0;
% the current best solution
fmin = min(sample_y);
% print the current information to the screen
fprintf('Pseuso EI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation < max_evaluation 
    % build the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    infill_x = zeros(num_q,num_vari);
    point_added = [];
    for ii = 1: num_q
        % find the point with the highest pseudo EI value using GA algorithm
        infill_x(ii,:) = Optimizer_GA(@(x)-Infill_PEI(x,GP_model,fmin,point_added),num_vari,lower_bound,upper_bound,10*num_vari,200);
        % update point_added
        point_added = infill_x(1:ii,:);
    end
    % evaluate the query points with the real function 
    best_y = feval(fun_name,infill_x);
    % add the new points to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;best_y];
    % update some parameters
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    fmin = min(sample_y);
    % print the current information to the screen
    fprintf('Pseuso EI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end




