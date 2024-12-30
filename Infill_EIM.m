function y = Infill_EIM(x,kriging_obj,non_dominated_front,criterion)
% you can choose criterion as 'Euclidean', 'Maximin', or 'Hypervolume'
num_x = size(x,1);
% number of non-dominated points,number of objectives
[num_pareto,num_obj] = size(non_dominated_front);
% the kriging prediction and varince
u = zeros(num_x,num_obj);
s = zeros(num_x,num_obj);
for ii = 1 : num_obj
    [u(:, ii),s(:, ii)] = GP_Predict(x, kriging_obj{ii});
end
u_matrix = repelem(u,num_pareto,1);
s_matrix = repelem(s,num_pareto,1);
f_matrix = repmat(non_dominated_front,num_x,1);
EIM = (f_matrix - u_matrix).*normcdf((f_matrix - u_matrix)./s_matrix) + s_matrix.*normpdf((f_matrix - u_matrix)./s_matrix);
switch criterion
    case 'Euclidean'
        y = min(reshape(sqrt(sum(EIM.^2,2)),[num_pareto,num_x]))';
    case 'Maximin'
        y = min(reshape(max(EIM,[],2),[num_pareto,num_x]))';
    case 'Hypervolume'
        ref_point = 1.1*ones(1, num_obj);
        y = min(reshape(prod(ref_point-f_matrix+EIM,2)-prod(ref_point-f_matrix,2),[num_pareto,num_x]))';
end
