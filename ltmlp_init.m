function net = ltmlp_init(net, opt, input)

errstring = ltmlp_consist(net, 'ltmlp network', size(input, 1));
if ~isempty(errstring)
  error(errstring);
end

nlayers = numel(net.layers);
layers = net.layers;
net.options = opt;
net.num_params = 0;

% heuristic initialization
for l = 2:nlayers,
  net.bias{l} = opt.bias_init_scale * (2 * rand(layers(l), 1) - 1);

  for ll=1:l-1,
    % Initialize shortcut connections to zero
    if ll < l - 1
      net.W{l, ll} = zeros(layers(l), layers(ll));
    else
      net.W{l, ll} = sqrt(6 / (net.layers(l) + net.layers(ll))) ...
        * (2 * rand(net.layers(l), net.layers(ll)) - 1);
      % (LeCun, 1993):
      % net.W{l, ll} = sqrt(1 / (net.layers(l))) ...
      %  * randn(net.layers(l), net.layers(ll));
    end
    net.num_params = net.num_params + layers(l) * layers(ll);    
  end
end

% AUTOENCODER: Remove shortcuts that skip the bottleneck
if strcmpi(opt.task, 'autoencoder')
  if nlayers==3 || nlayers==4
    net.num_params = net.num_params - numel(net.W(3,1));
    net.W(3,1) = cell(1,1);
  elseif nlayers==5 || nlayers==6
    for l = 4:5
      for ll = 1:2
        net.num_params = net.num_params - numel(net.W(l,ll));
      end
    end
    net.W(4:5,1:2) = cell(2,2);
  elseif nlayers==7 || nlayers==8
    for l = 5:7
      for ll = 1:3
        net.num_params = net.num_params - numel(net.W(l,ll));
      end
    end
    net.W(5:7,1:3) = cell(3,3);
  elseif nlayers==9
    for l = 6:9
      for ll = 1:4
        net.num_params = net.num_params - numel(net.W(l,ll));
      end
    end
    net.W(6:9,1:4) = cell(4,4);
  end
end

% initialize transformed nonlinearities
net.nonlintrans = cell(1,nlayers);
for l=2:nlayers
  if numel(opt.fixed_transf) > 1
    net.nonlintrans{l} = repmat(opt.fixed_transf, [layers(l) 1]);
  else % this equals fixedtransf [0 0 1] (= no transformations)
    net.nonlintrans{l} = [zeros(layers(l), 2) ones(layers(l), 1)];
  end
end