%--------------------------------------------------------------------------
% This is the multi-point expected improvement approach for batch Bayesian
% optimization. The qEI function is coded according to the R code in [1]. I
% used the Modified Cholesky algorithm [2] and a quasi-random approach to
% estimate  an MVN probability [3] in the qEI implementation.
% Reference: 
% [1] O. Roustant, D. Ginsbourger, and Y. Deville. DiceKriging,
%   DiceOptim: Two R Packages for the Analysis of Computer Experiments by
%   Kriging-Based Metamodeling and Optimization. Journal of Statistical
%   Software, 2012, 51(1): 1-55.
% [2] S. H. Cheng and N. J. Higham. A modified Cholesky algorithm based
%   on a symmetric indefinite factorization. SIAM J. Matrix Anal. Appl.,
%   19(4):1097-1110, 1998. https://github.com/higham/modified-cholesky.
% [3] Alan Genz. Numerical Computation of Multivariate Normal
% Probabilities. J. of Computational and Graphical Stat., 1992 1: 141-149.
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
fprintf('qEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
% the iteration
while evaluation < max_evaluation
    % build the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    % maximize the qEI function
    [best_x,max_EI] = Optimizer_GA(@(x)-Infill_qEI(x,GP_model,fmin),num_vari*num_q,repmat(lower_bound,1,num_q),repmat(upper_bound,1,num_q),num_vari*num_q,100);
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
    % print the current information to the screen
    fprintf('qEI on %d-D %s function, iteration: %d, evaluation: %d, current best solution: %0.2f\n',num_vari,fun_name,iteration,evaluation,fmin);
end

