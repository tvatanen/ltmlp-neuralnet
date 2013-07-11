function [net, res] = ltmlp_train2(net, input, output, test_input, test_output)

% Training algorithm with burn-in (increasing momentum, exponentially
% decaying learning rate). This is specifically for classification task.
% Shows error in number of items classified incorrecly.
% Runtime (opt.runtime) in epochs.

opt = net.options;
momentum = opt.mominit;
nlayers = numel(net.layers);
layers = net.layers;
nonlintypes = net.layertypes;
gradW = cell(nlayers,nlayers-1);
gradbias = cell(nlayers,1);
stepsizeW = zeros(nlayers,nlayers-1);
gammareco = cell(1,nlayers);
stepsize = opt.stepsize;

% Stepsize for shortcut connections is smaller
for l=2:nlayers,
  for ll=1:l-1,
    stepsizeW(l,ll) = 1/2^(l-ll-1);
  end
end

% Initialize momentum to zero
if momentum > 0
  directionW = cell(nlayers,nlayers-1);
  directionbias = cell(nlayers,1);
  for l=1:nlayers,
    directionbias{l} = zeros(layers(l), 1);
    for ll=1:l-1,
      directionW{l,ll} = zeros(size(net.W{l,ll}));
    end
  end
end

if nargout > 1 || opt.verbose
  save_data = 1;
  res.training_errors = ones(1,opt.runtime)*Inf;
  res.test_errors = ones(1,opt.runtime)*Inf;
  res.iters = ones(1,opt.runtime)*Inf;
  res.gradW = ones(1,opt.runtime)*Inf;
  res.cputimes = ones(1,opt.runtime)*Inf;
  num_evals_complete = 0;
else
  save_data = 0;
end

[datadim, dlen] = size(input);
blen = opt.minibatchsize;
i0 = 0;

iter = 1;
cpustart = cputime;
epoch = 0;

while epoch < opt.runtime
  
  % Choose mini batch
  inds = (i0+1):min(dlen,i0+blen); 
  i0 = i0+blen;
  if i0 >= dlen, i0 = 0;end
  
  % Add noise to activations for regularization
  if opt.input_noise > 0
    current_input = input(:,inds) + opt.input_noise * randn(datadim, numel(inds));
  else
    current_input = input(:,inds);
  end

  % Update transformations
  if opt.num_transf > 0
    if num_evals_complete == 0 || res.test_errors(num_evals_complete) > 100
      net = ltmlp_transform(net, current_input);
    elseif mod(blen*iter,dlen) == 0
      net = ltmlp_transform(net, input + opt.input_noise * randn(datadim, size(input,2)));
    end
  end
  
  % Feedforward
  [current_output, net] = ltmlp_ff(net, current_input); 
    
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

  % Update momentum
  if momentum > 0
    for l=2:nlayers
      directionbias{l} = gradbias{l} + opt.momentum * directionbias{l};
      % update rule from dropout paper:
      %directionbias{l} = momentum * directionbias{l} - ...
      %  (1 - momentum) * stepsize * stepsizebias(l) * gradbias{l};
      for ll=1:l-1,
        if ~isempty(gradW{l,ll})
          directionW{l,ll} = gradW{l,ll} + momentum * directionW{l,ll};
          % update rule from dropout paper:
          %directionW{l,ll} = momentum * directionW{l,ll} - ...
          %  (1 - momentum) * stepsize * stepsizeW(l,ll) * gradW{l,ll};
        end
      end
    end
  end
  
  % Update weights
  for l = 2:nlayers
    if momentum > 0
      net.bias{l} = net.bias{l} - stepsize * directionbias{l};
      % update rule from dropout paper
      %net.bias{l} = net.bias{l} + directionbias{l};
    else
      net.bias{l} = net.bias{l} - stepsize * gradbias{l};
    end
    for ll=1:l-1,
      if (~isempty(net.W{l,ll}))
        if momentum > 0
          net.W{l,ll} = net.W{l,ll} - stepsize * stepsizeW(l,ll) * directionW{l,ll};
          % update rule from dropout paper:          
          % net.W{l,ll} = net.W{l,ll} + directionW{l,ll};
        else
          net.W{l,ll} = net.W{l,ll} - stepsize * stepsizeW(l,ll) * gradW{l,ll};
        end        
      end
    end
  end
  iter = iter + 1;

  % Error calculations
  if mod(blen*iter,dlen) == 0 % && save_data 
    epoch = epoch+1;

    % Exponentially decreasing stepsize
    % linearly increasing momentum
    if epoch >= opt.burnin
      stepsize = stepsize * opt.ratedecay;
      momentum = opt.mominit;
    else
      momentum = (epoch/opt.burnin)*opt.momentum + (1-(epoch/opt.burnin))*opt.mominit;
    end
    
    if save_data
      cputemp = cputime;      
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
        res.training_errors(num_evals_complete) = sum(maxI1~=maxI2);
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
        res.test_errors(num_evals_complete) = sum(maxI1~=maxI2);
      else
        res.test_errors(num_evals_complete) = NaN;
      end
      if momentum > 0
        res.gradW(num_evals_complete) = sum(ltmlp_params2vector(directionW, net.num_params).^2);
      else
        res.gradW(num_evals_complete) = sum(ltmlp_params2vector(gradWm, net.num_params).^2);
      end

      if opt.verbose
        fprintf('Epoch %4d: training error = %d, test error = %d, cputime = %d, grad = %.4f\n', ...
          epoch, res.training_errors(num_evals_complete), res.test_errors(num_evals_complete), ...
          floor(cputime-cpustart), res.gradW(num_evals_complete));
      end
      cpustart = cpustart + cputime - cputemp;
    end
  end
end