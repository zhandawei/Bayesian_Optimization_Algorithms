% This is the matlab implementation of the Bayesian optimization algorithm
% according to the following work. 
% We use a real-coded genetic algorithm to maximize the EI function.  
% D. R. Jones, M. Schonlau, and W. J. Welch. Efficient global optimization
% of expensive black-box functions. Journal of Global Optimization, 1998,
% 13(4): 455-492.
% author: Dawei Zhan
% date: 2024/11/26
clearvars; close all;
% setting of the problem
fun_name = 'Rosenbrock';
num_vari = 10;
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% the number of initial design points
num_initial = 20;
% maximum number of evaluations
max_evaluation = 100;
% initial design points using Latin hypercube sampling method
sample_x = lower_bound + (upper_bound-lower_bound).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name,sample_x);
% the current iteration and evaluation
evaluation = size(sample_x,1);
iteration = 0;
% the current best solution
fmin = min(sample_y);
% print the current information to the screen
fprintf('BO on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation <  max_evaluation
    % build the Kriging model
    kriging_model = Kriging_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % find the point with the highest EI value using GA algorithm
    infill_x = Optimizer_GA(@(x)-Infill_EI(x,kriging_model,fmin),num_vari,lower_bound,upper_bound,50,100);
    % evaluating the query point with the real function
    infill_y = feval(fun_name,infill_x);
    % add the new point to design set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update some parameters
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    fmin = min(sample_y);
    % print the current information to the screen
    fprintf('BO on %d-D %s function, iteration: %d, evaluation: %d, current found minimum: %f\n',num_vari,fun_name,iteration,evaluation,fmin);
end



