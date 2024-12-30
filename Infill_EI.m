function obj = Infill_EI(x,model,fmin)
% get the GP prediction and variance
[u,s] = GP_Predict(x,model);
% calcuate the EI value
EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
% this EI needs to be maximized
obj = EI;

end





