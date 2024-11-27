function obj = Infill_PEI(x,kriging_model,fmin,point_added)
% the pseudo EI criterion
% get the Kriging prediction and variance
[u,s] = Kriging_Predictor(x,kriging_model);
% calcuate the EI value
EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
lower_bound = kriging_model.lower_bound;
upper_bound = kriging_model.upper_bound;
theta = kriging_model.theta;
% if this is not the first infill point
if ~isempty(point_added)
    % the scaling of x
    x1 = (x - lower_bound)./(upper_bound- lower_bound);
    x2 = (point_added - lower_bound)./(upper_bound - lower_bound);
    % calculate the correlation
    temp1 = sum(x1.^2.*theta,2)*ones(1,size(x2,1));
    temp2 = sum(x2.^2.*theta,2)*ones(1,size(x1,1));
    correlation = exp(-(temp1 + temp2'-2.*(x1.*theta)*x2'));
    % the Pseudo EI matrix
    EI = EI.*prod(1-correlation,2);
end
% the objective needs to be maximized
obj = EI;
end





