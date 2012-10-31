function param_vector = ltmlp_params2vector(params, numparams)

param_vector = zeros(numparams,1);
w_count = 1;
for l = 2:size(params,1)
  for ll=1:l-1,
    if ~isempty(params{l,ll})
      params_tmp = params{l,ll}(:);
      param_vector(w_count:w_count + length(params_tmp) - 1) = params_tmp;
      w_count = w_count + length(params_tmp);
    end
  end
end