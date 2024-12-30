%--------------------------------------------------------------------------
% This is the Kriging believe approach which always use the Kriging
% prediction value as the fake objective to update the GP model to produce
% multiple query points for parallel function evaluations. It is coded
% based on the following work.
% Reference: 
% D. Ginsbourger, R. Le Riche, and L. Carraro. Kriging is
% well-suited to parallelize optimization. Computational intelligence in
% expensive optimization problems. 2010, 131-162.
% Author: Dawei Zhan
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
% batch size
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
fprintf('Kriging Believer on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation < max_evaluation
    % train the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    infill_x = zeros(num_q,num_vari);
    for ii = 1: num_q
        [infill_x(ii,:),max_EI] = Optimizer_GA(@(x)-Infill_EI(x,GP_model,fmin),num_vari,lower_bound,upper_bound,10*num_vari,200);
        % use GP prediction as the fake objective
        GP_model = GP_Train([sample_x;infill_x(1:ii,:)],[sample_y;GP_Predict(infill_x(1:ii,:),GP_model)],lower_bound,upper_bound,GP_model.theta);
    end
    % evaluate the query points with the real function
    infill_y = feval(fun_name,infill_x);
    % add the new points to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update some parameters
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    fmin = min(sample_y);
    % print the current information to the screen
    fprintf('Kriging Believer on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end




