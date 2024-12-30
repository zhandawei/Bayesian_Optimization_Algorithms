function y = Infill_ESSI(x,model,fmin,best_x,subspace)

n = size(x,1);
new_x = repmat(best_x,n,1);
new_x(:,subspace) = x;
[u,s] = GP_Predict(new_x,model);
y = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
end
