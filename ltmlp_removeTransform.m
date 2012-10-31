function net = ltmlp_removeTransform(net, input)

% transform nonlinear functions such that both the average output and
% average derivative of the output are zeroes

nlayers = numel(net.layers);
X = cell(nlayers,1);
Y = cell(nlayers,1);
bias = net.bias;
W = net.W;
nonlintypes = net.layertypes;
nonlintrans = net.nonlintrans;
dlen = size(input,2);
opt = net.options;

Y{1} = input;

if opt.verbose == 4
  % contains variances of f(b_i x_t) and f'(b_i x_t) for each layer
  nonlinoutput_vars = ones(nlayers,2)*Inf;
end

for l = 2:nlayers

  X{l} = repmat(bias{l}, [1 dlen]);
  
  for ll = 1:l-1
    if ~isempty(W{l,ll})
      X{l} = X{l} + W{l,ll} * Y{ll};
    end
  end
  if any([strcmp(nonlintypes{l-1},'tanh')  strcmp(nonlintypes{l-1},'softsign') strcmp(nonlintypes{l-1},'rect')])
    oldtrans = nonlintrans{l};
    nonlintrans{l}(:,1) = ones(size(oldtrans,1),1);
    if opt.num_transf > 1
      nonlintrans{l}(:,2) = zeros(size(oldtrans,1),1);
    end
    
    % Compensate alpha and beta by updating the shortcut weights
    for lll = l+1:nlayers
      if ~isempty(W{lll,l})
        bias{lll} = bias{lll} + W{lll,l}*(bsxfun(@times,(oldtrans(:,1)-nonlintrans{l}(:,1)).*nonlintrans{l}(:,3),bias{l}) + (oldtrans(:,2)-nonlintrans{l}(:,2)).*nonlintrans{l}(:,3));
        for ll = 1:l-1
          if ~isempty(W{lll,ll})
            W{lll,ll} = W{lll,ll} + bsxfun(@times,W{lll,l},((oldtrans(:,1)-nonlintrans{l}(:,1)).*nonlintrans{l}(:,3))') * W{l,ll};
          end
        end
      end
    end
  end
  
  Y{l} = nonlin(X{l}, nonlintypes{l-1}, nonlintrans{l});

end

if opt.num_transf > 2
  for l = 2:nlayers

    X{l} = repmat(bias{l}, [1 dlen]);

    for ll = 1:l-1
      if ~isempty(W{l,ll})
        X{l} = X{l} + W{l,ll} * Y{ll};
      end
    end
    if any([strcmp(nonlintypes{l-1},'tanh')  strcmp(nonlintypes{l-1},'softsign') strcmp(nonlintypes{l-1},'rect')])
      oldtrans = nonlintrans{l}(:,3);
      
      nonlintrans{l}(:,3) = ones(size(oldtrans,1),1);
      
      % Compensate for gamma:
      for lll = l+1:nlayers
        if ~isempty(W{lll,l})
          W{lll,l} = bsxfun(@times, W{lll,l}, repmat(bsxfun(@rdivide, oldtrans, nonlintrans{l}(:,3))', [size(W{lll,l},1) 1]));
        end
      end
    end
    
    Y{l} = nonlin(X{l}, nonlintypes{l-1}, nonlintrans{l});

    if opt.verbose == 4
      df = nonlin(X{l}, nonlintypes{l-1}, nonlintrans{l},1);
      nonlinoutput_vars(l,:) = [var(Y{l}(:)) var(df(:))];
    end
  end
end
  
if opt.verbose == 4
  fprintf('\nVariances of f() and df() on layer ')
  for l = 2:nlayers
    if any([strcmp(nonlintypes{l-1},'tanh')  strcmp(nonlintypes{l-1},'softsign') strcmp(nonlintypes{l-1},'rect')])
      fprintf('%d: %.2f (%.2f), ', l, nonlinoutput_vars(l,1), nonlinoutput_vars(l,2));
    end
  end
  fprintf('\n')
end

net.X = X;
net.Y = Y;
net.bias = bias;
net.W = W;
net.nonlintrans = nonlintrans;