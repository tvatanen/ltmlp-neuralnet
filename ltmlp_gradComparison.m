function [net res] = ltmlp_gradComparison(net, input, output, test_input, test_output)

nlayers = numel(net.layers);
nonlintypes = net.layertypes;
opt = net.options;
% Y = net.Y;
% X = net.X;
% W = net.W;
% bias = net.bias;

W_init = net.W;
%Mu_list = [0.5 0.4 0.3 0.2 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.0001];
Mu_list = logspace(-4, -0.3, 20);

gradW = cell(nlayers,nlayers-1);
gradbias = cell(nlayers,1);
stepsizeW = zeros(nlayers,nlayers-1);
stepsizebias = ones(nlayers,1);
gammareco = cell(1,nlayers);
stepsize = opt.stepsize;

% Initial stepsizes
% Stepsizes of shortcut connections are smaller:
for l=2:nlayers,
  for ll=1:l-1,
    stepsizeW(l,ll) = 1/2^(l-ll-1);
  end
end

if nargout > 1 || opt.verbose
  save_data = 1;
  res.training_errors = ones(1,opt.n_error_evals)*Inf;
  res.test_errors = ones(1,opt.n_error_evals)*Inf;
  res.iters = ones(1,opt.n_error_evals)*Inf;
  res.gradW = ones(1,opt.n_error_evals)*Inf;
  res.cputimes = ones(1,opt.n_error_evals)*Inf;
  res.thetas = ones(length(Mu_list),30)*Inf;
  res.eigs = ones(net.num_params,30);
  res.hessian_diag = ones(net.num_params,30);
  num_thetas_computed = 0;
  num_hessians_computed = 0;
  num_evals_complete = 0;
else
  save_data = 0;
end

[datadim dlen] = size(input);
blen = opt.minibatchsize;
i0 = 0;

iter = 1;
cpustart = cputime;

while cputime - cpustart < opt.runtime
  
  % Decreased stepsize with autoencoder in the beginning of the training 
  % to avoid early divergence
%   if iter < 1000
%     stepsize = 100^(iter/100-1)*opt.stepsize;
%   end
  
  
  % Choose mini batch
  inds = (i0+1):min(dlen,i0+blen); 
  i0 = i0+blen;
  if i0 >= dlen, i0 = 0;end
  
  % Add noise to avoid over-learning 
  if opt.input_noise > 0
    current_input = input(:,inds) + opt.input_noise * randn(datadim, numel(inds));
  else
    current_input = input(:,inds);
  end

  
  % Update transformations
  if opt.num_transf > 0 && mod(iter,opt.transf_every_n_iters)==0 && opt.fixed_transf(1) == 0
    % discount the time used updating transformations
    %cputemp = cputime; 
    
    if opt.verbose == 4
      [current_output_before, net] = ltmlp_ff(net, current_input); 
    end
    if opt.verbose
      fprintf('Updating transformations with full batch...');
    end
    net = ltmlp_transform(net, input);
    
    if opt.verbose
      fprintf('done.\n');
    end
    %cpustart = cpustart + cputime - cputemp;
  end
  % Feedforward
  [current_output, net] = ltmlp_ff(net, current_input); 
  
  % Show the effect of the transformation to the output (for debugging)
  if mod(iter,opt.transf_every_n_iters)==0 && opt.verbose == 4 && opt.fixed_transf(1) == 0
    fprintf('Effect of the transformation to output = %.4f\n', ...
      sum((current_output(:)-current_output_before(:)).^2));
  end
  
  % Compute reconstruction error
  reco_err = current_output - output(:,inds);

  % Backpropagation part
  gammareco{nlayers} = nonlin(net.X{nlayers}, nonlintypes{nlayers-1}, net.nonlintrans{nlayers}, 1) .* reco_err;
  for l = nlayers-1:-1:2,
    gammareco{l} = 0;
    for ll=(l+1):nlayers,
      if ~isempty(net.W{ll,l})
        gammareco{l} = gammareco{l} + (net.W{ll,l}' * gammareco{ll}) .* ...
          nonlin(net.X{l}, nonlintypes{l-1}, net.nonlintrans{l}, 1);
      end
    end
  end
  
  % Gradient computations
  for l = 2:nlayers
    % Regular gradient
    gradbias{l} = sum(gammareco{l}, 2) / numel(inds);
    for ll=1:l-1,
      if ~isempty(net.W{l,ll})
        gradW{l,ll} = gammareco{l} * net.Y{ll}' / numel(inds);
      end
    end
    if opt.weight_decay>0
      % Weight decay
      gradbias{l} = gradbias{l} + opt.weight_decay * net.bias{l};
      for ll=1:l-1,
        if ~isempty(net.W{l,ll})
          gradW{l,ll} = gradW{l,ll} + opt.weight_decay * net.W{l,ll};
        end
      end
    end
  end
  
  %% Compare different update vectors
  
  %if mod(iter,opt.transf_every_n_iters/2)==0
  if iter == 100 || iter == 4000 || iter == 8000
    cputemp = cputime;    
    if opt.verbose
      fprintf('Approximating Hessian matrix...');
    end
    %Hessian = ltmlp_hessian(net, current_input, output(:,inds), gradW);
    Hessian = ltmlp_hessian(net, input, output, gradW);
    % symmetrize
    Hessian = (Hessian+Hessian')/2;
    % store eigenvalues
    num_hessians_computed = num_hessians_computed + 1;
    res.eigs(:,num_hessians_computed) = eig(Hessian);
    res.hessian_diag(:,num_hessians_computed) = diag(Hessian);
    
    if opt.verbose
      fprintf('done\n');
    end
    
    % Concatenate gradient into a vector
    gradVector = ltmlp_params2vector(gradW, net.num_params);

    % Update direction of the second order method
    if opt.verbose
      fprintf('Inverting Hessian matrix...');
    end
    num_thetas_computed = num_thetas_computed + 1;    
    for mu_idx = 1:length(Mu_list)
      
      % check if the smallest eigenvalue of regularized Hessian is positive
      eigs = eig(Hessian+eye(size(Hessian,1))*Mu_list(mu_idx));
      if eigs(1) > 1e-5
        secondOrderUpdate = (Hessian+eye(size(Hessian,1))*Mu_list(mu_idx)) \ gradVector;

        % Compute the angle between gradients
        costheta = dot(gradVector,secondOrderUpdate)/(norm(gradVector)*norm(secondOrderUpdate));
        res.thetas(mu_idx,num_thetas_computed) = acos(costheta);
      else
        res.thetas(mu_idx,num_thetas_computed) = NaN;
      end
      %fprintf('Angle between second order and grad: %.4f\n', ...
      %  res.thetas(num_thetas_computed));
    end

    if opt.verbose
      fprintf('done\n');
    end

    % Discount the time spent evaluating angles
    cpustart = cpustart + cputime - cputemp;
    
  end
  
  % Update momentum
  % Momentum is NOT USED in gradient angle comparison
%   for l=2:nlayers
%     directionbias{l} = gradbias{l} + opt.momentum_grad * directionbias{l};
%     for ll=1:l-1,
%       if ~isempty(gradW{l,ll})
%         directionW{l,ll} = gradW{l,ll} + opt.momentum_grad * directionW{l,ll};
%       end
%     end
%   end
  
  % Update weights
  for l = 2:nlayers
    net.bias{l} = net.bias{l} - stepsize * stepsizebias(l) * gradbias{l};
    for ll=1:l-1,
      if (~isempty(net.W{l,ll}))
        net.W{l,ll} = net.W{l,ll} - stepsize * stepsizeW(l,ll) * gradW{l,ll};
      end
    end
  end
  
  prop_time_used = (cputime-cpustart)/opt.runtime;

  % Decrease stepsize in the beginning of autoencoder training
  if strcmp(opt.task, 'autoencoder') && prop_time_used <0.01
    stepsize = 100^(prop_time_used/0.01-1)*opt.stepsize;

  % Decrease stepsize during the second half of the training
  elseif prop_time_used > 0.5
    stepsize = max(1e-8,(1-prop_time_used)*2*opt.stepsize);
  end
  
  % Error calculations
  if save_data && prop_time_used > (num_evals_complete+1)/opt.n_error_evals
    cputemp = cputime;
    
    num_evals_complete = num_evals_complete + 1;
    res.iters(num_evals_complete) = iter;
    res.cputimes(num_evals_complete) = cputime-cpustart;
    
    [train_error, test_error] = ltmlp_error(net, input, output, test_input, test_output);
    res.training_errors(num_evals_complete) = train_error;
    res.test_errors(num_evals_complete) = test_error;
    res.gradW(num_evals_complete) = sum(w2vec(gradW).^2);

    if opt.verbose
      fprintf('Iteration %4d: training error = %.4f, test error = %.4f, cputime = %d, grad = %.4f\n', ...
        iter, res.training_errors(num_evals_complete), ...
        res.test_errors(num_evals_complete), ...
        floor(cputime-cpustart), res.gradW(num_evals_complete));
    end
    
    %if isnan(res.training_errors(num_evals_complete)), break;end
    
    if opt.verbose == 2;
      for l = 2:nlayers
        fprintf('Norms of update on layer %d: ',l);
        for ll=1:l-1,
          if (~isempty(directionW{l,ll}))
            fprintf('%f, ',sqrt(mean(mean(directionW{l,ll}.^2))));
          end
        end
        fprintf('\n');
      end
    end
    
    %       save(sprintf('run_cv%d_dt%d_sc%d_n%d_ss%f_no%f.mat',crossvalid,opt.datatype,opt.shortcuts,sum(nl),opt.initstepsize,opt.init_input_noise), ...
    %         'W','bias','res','opt','nl','data_mean','directionW','directionbias','nonlintrans','nonlintypes','pca_W',...
    %         'nlayers','stepsizeW','stepsizebias','iter','nsamples','W_init');
    
    % Plotting
    if opt.verbose == 2
      % visualize weights
      set(0, 'CurrentFigure', 2);
      visualize(net.W{2,1}',1);
      if (nlayers>2),
        if (~isempty(net.W{3,1})),
          set(0, 'CurrentFigure', 3);
          visualize(net.W{3,1}',1);
        end
      end
      if (nlayers>3),
        if (~isempty(net.W{4,1})),
          set(0, 'CurrentFigure', 4);
          visualize(net.W{4,1}',1);
        end
      end
      %         if (nlayers>4),
      %           if (~isempty(W{5,1})),
      %             set(0, 'CurrentFigure', 5);
      %             visualize(pca_W*W{5,1}',1+2*(opt.datatype==5));
      %           end
      %         end
      %         if (and(opt.datatype==4, opt.symmetrization==0))
      %           set(0, 'CurrentFigure', 12);
      %           visualize(pca_W*W{nlayers,nlayers-1});
      %           if (nlayers>2),
      %             if (~isempty(W{3,1})),
      %               set(0, 'CurrentFigure', 13);
      %               visualize(pca_W*W{nlayers,nlayers-2});
      %             end
      %           end
      %           if (nlayers>3),
      %             if (~isempty(W{4,1})),
      %               set(0, 'CurrentFigure', 14);
      %               visualize(pca_W*W{nlayers,nlayers-3});
      %             end
      %           end
      %         end
      set(0, 'CurrentFigure', 1);
      for l = 2:nlayers
        %            title('angles from initialization');
        subplot(nlayers-1,1,l-1);
        hist(acos(bsxfun(@rdivide, sum(net.W{l,l-1}.*W_init{l,l-1},2), sqrt(sum(net.W{l,l-1}.^2,2).*sum(W_init{l,l-1}.^2,2))))*180/pi,50);
      end
      set(0, 'CurrentFigure', 6);
      plot(blen*res.iters(floor(num_evals_complete/10)+1:end),res.test_errors(floor(num_evals_complete/10)+1:end));
      %         else
      %           plot(res.cputimes(floor(num_evals_complete/10)+1:end),[training_errors(floor(num_evals_complete/10)+1:end); test_errors(floor(num_evals_complete/10)+1:end)]');
      %           % legend('training error','test error');
      %         end
      
      drawnow;
      
    end
    
    % Discount the time spent validating
    cpustart = cpustart + cputime - cputemp;
    
  end
  iter = iter+1;
  
end

% Compute final errors 
num_evals_complete = num_evals_complete + 1;
res.iters(num_evals_complete) = iter;
res.cputimes(num_evals_complete) = cputime-cpustart;

% Training error
current_training_output = ltmlp_ff(net, input);

if strcmp(opt.task,'regression') || strcmp(opt.task,'autoencoder')
  res.training_errors(num_evals_complete) = mean(sum((current_training_output-output).^2,1),2);
elseif strcmp(opt.task,'classification')
  [~,maxI1] = max(current_training_output);
  [~,maxI2] = max(output);
  res.training_errors(num_evals_complete) = 100 - 100*sum(maxI1==maxI2)/length(maxI1);
else
  res.training_errors(num_evals_complete) = NaN;
end

% Test error 
current_test_output = ltmlp_ff(net, test_input);
if strcmp(opt.task,'regression') || strcmp(opt.task,'autoencoder')
  res.test_errors(num_evals_complete) = mean(sum((current_test_output-test_output).^2,1),2);
elseif strcmp(opt.task,'classification')
  [~,maxI1]=max(current_test_output);
  [~,maxI2]=max(test_output);
  res.test_errors(num_evals_complete) = 100 - 100*sum(maxI1==maxI2)/length(maxI1);
else
  res.test_errors(num_evals_complete) = NaN;
end
res.gradW(num_evals_complete) = sum(w2vec(gradW).^2);

if opt.verbose
  fprintf('Training done : training error = %.4f, test error = %.4f, cputime = %d, grad = %.4f\n', ...
    res.training_errors(num_evals_complete), ...
    res.test_errors(num_evals_complete), ...
    floor(cputime-cpustart), res.gradW(num_evals_complete));
end

