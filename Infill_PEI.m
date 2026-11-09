function y = Infill_PEI(new_x,model,fmin,point_added)
% the pseudo EI criterion
lower_bound = model.lower_bound;
upper_bound = model.upper_bound;
% get the Kriging prediction and variance
[u,s] = GP_Predict(new_x,model);
% calcuate the EI value
EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
% if this is not the first infill point
if ~isempty(point_added)
    theta = model.theta;
    % normalize data
    X = (new_x - lower_bound)./(upper_bound- lower_bound);
    x = (point_added - lower_bound)./(upper_bound - lower_bound);
    % the correlation
    temp1 = sum(x.^2.*theta,2)*ones(1,size(X,1));
    temp2 = sum(X.^2.*theta,2)*ones(1,size(x,1));
    corr = exp(-(temp1 + temp2'-2.*(x.*theta)*X'))';
    % the Pseudo EI 
    EI = EI.*prod(1-corr,2);
end
% the objective needs to be maximized
y = EI;
end





