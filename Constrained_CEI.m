clearvars;close all;
% settings of the problem
fun_name = 'Fun_Welded_Beam';
num_vari = 4;    
num_con = 6;  
lower_bound = [0.1, 0.1, 0.1, 0.1];
upper_bound = [2.0, 10.0, 10.0, 2.0];
% the number of initial design points
num_initial = 20;
% maximum number of evaluations
max_evaluation = 100;
% the 0th iteration
% initial design points using Latin hypercube sampling method
sample_x = lower_bound + (upper_bound-lower_bound).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
[sample_y, sample_g] = feval(fun_name, sample_x);
% the number of total evaluations
evaluation = size(sample_x,1);
iteration = 0;
% check is there is at least one feasible solution
index = sum(sample_g <= 0, 2) == num_con;
if sum(index) ~= 0
    fmin = min(sample_y(index, :));
    fprintf('iteration: %d, evaluation: %d, best solution: %f\n', 0, evaluation, fmin);
else
    fmin  = 1E6;
    fprintf('iteration: %d, evaluation: %d, best solution: no feasiable solution\n', 0, evaluation);
end
% beginning of the iteration
while evaluation < max_evaluation
    % if there is no feasiable solution, there is no need to build the
    % kriging model for the objective function (build anyway)
    kriging_con = cell(1,num_con);
    for ii = 1: num_con
        kriging_con{ii} = Kriging_Train(sample_x,sample_g(:, ii),lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    end
    kriging_obj = Kriging_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % select the infill samples
    if sum(index) ~= 0
        infill_criterion = @(x)-Infill_CEI(x,kriging_obj,kriging_con,fmin);
    else
        infill_criterion = @(x)-Infill_PoF(x,kriging_con);
    end
    % find the candidate by maximizing the infill criterion
    [best_x,max_EI] = Optimizer_GA(infill_criterion,num_vari,lower_bound,upper_bound,50,100);
    % evalaute the candidate points in parallel
    [best_y, best_g] = feval(fun_name,best_x);
    % add the new point to design set
    sample_x = [sample_x;best_x];
    sample_y = [sample_y;best_y];
    sample_g = [sample_g;best_g];
    % update the best solution
    % check is there is at least one feasible solution
    evaluation = size(sample_x,1);
    iteration = iteration + 1;
    index = sum(sample_g <= 0, 2) == num_con;
    if sum(index) ~= 0
        fmin = min(sample_y(index, :));
        fprintf('iteration: %d, evaluation: %d, best solution: %f\n',iteration,evaluation,fmin);
    else
        fmin = 1E6;
        fprintf('iteration: %d, evaluation: %d, best solution: no feasiable solution\n',iteration,evaluation);
    end
end





