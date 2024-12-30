function y = Infill_FqEI(x,model,fmin)
y = zeros(size(x,1),1);
for ii = 1:size(x,1)
    new_x = x(ii,:);
    % reshape the input
    new_x = reshape(new_x,size(model.sample_x,2),[])';
    % remove reduplicative points
    new_x = unique(new_x,'rows');
    % remove sampled points
    Lia = ismember(new_x,model.sample_x,'rows');
    new_x = new_x(~Lia,:);
    % number of infill samples
    q = size(new_x,1);
    % predictions and covarince matrix
    [u,s,Corr] = GP_Predict(new_x,model);
    EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
    y(ii) = sum(EI) - sum((sum(Corr.*min(EI*ones(1,q),ones(q,1)*EI'),2) - EI)/q);
end