function EHVI = Infill_EHVI(x,model,f)
num_x = size(x,1);
% number of objectives
num_obj = size(f,2);
r = 1.1*ones(1,num_obj);
% the kriging prediction and varince
u = zeros(num_x,num_obj);
s = zeros(num_x,num_obj);
for ii = 1:num_obj
    [u(:, ii),s(:, ii)] = GP_Predict(x,model{ii});
end
EHVI = zeros(num_x,1);
for ii = 1:size(x,1)
    % EHVI calculated using Monte Carlo simulation
    num_simluation_point = 50;
    num_simultion_HV = 50;
    hvi = zeros(num_simluation_point,1);
    rand_sample = mvnrnd(u(ii,:),diag(s(ii,:).^2),num_simluation_point);
    for jj = 1:num_simluation_point
        if any(all(f <= rand_sample(jj,:),2)) % dominated solution
            hvi(jj) = 0;
        else % non dominated solution
            new_front = [f;rand_sample(jj,:)];
            upper_bound = r;
            lower_bound = min(new_front);
            sim_point = rand(num_simultion_HV,num_obj).*(upper_bound-lower_bound)+lower_bound;
            num_front_point = size(new_front,1);
            simulated_point_matrix = repelem(sim_point,num_front_point,1);
            new_front_matrix = repmat(new_front,num_simultion_HV,1);
            is_dominated = reshape(all(new_front_matrix <= simulated_point_matrix,2),num_front_point,num_simultion_HV)';
            num_improvement = sum(any(is_dominated,2)) - sum(any(is_dominated(:,1:size(f,1)),2));
            hvi(jj) = prod(upper_bound-lower_bound)*num_improvement/num_simultion_HV;
        end
    end
    EHVI(ii,:) = mean(hvi);
end
