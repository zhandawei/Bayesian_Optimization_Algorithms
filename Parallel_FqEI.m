%-------------------------------------------------------------------------
% This is the matlab implementation of the fast multi-point expected
% improvement approach. The FqEI criterion has very similar properties as
% the qEI, but is significantly faster than qEI. Therefore, it can be used
% for large batch size.
% Reference: 
% D. Zhan, Y. Meng, and H. Xing. A Fast Multipoint Expected Improvement 
% for Parallel Expensive Optimization. IEEE Transactions on Evolutionary 
% Computation, 2023, 27(1): 170:184.
% Author: Dawei Zhan
% Date: 2024.11.27
%-------------------------------------------------------------------------
clearvars; close all;
% setting of the problem
fun_name = 'Rosenbrock';
num_vari = 10;
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% number of initial design points
num_initial = 20;
% maximum number of evaluations
max_evaluation = 120;
% batch size
num_q = 4;
% initial design points using Latin hypercube sampling method
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound) + lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation = size(sample_x,1);
iteration = 0;
% current best solution
fmin = min(sample_y);
% print the current information to screen
fprintf('FqEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation < max_evaluation
    % build the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % maximize the FqEI function using GA
    [best_x,max_EI] = Optimizer_GA(@(x)-Infill_FqEI(x,GP_model,fmin),num_vari*num_q,repmat(lower_bound,1,num_q),repmat(upper_bound,1,num_q),4*num_vari*num_q,200);
    infill_x = reshape(best_x,num_vari,[])';
    % evaluate the query points with the real function
    infill_y = feval(fun_name,infill_x);
    % add the new points to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update some parameters
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    fmin = min(sample_y);
    % print the current information to screen
    fprintf('FqEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end




