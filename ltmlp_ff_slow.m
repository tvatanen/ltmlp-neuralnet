function output = ltmlp_ff_slow(net, input, W_slow, bias_slow)

% Feedforward computations for ltmlp
net.W = W_slow;
net.bias = bias_slow;
net.Y{1} = input;

for l = 2:numel(net.layers)
  net.X{l} = repmat(net.bias{l}, 1, size(net.Y{1}, 2));
  for ll = 1:l-1,
    if (~isempty(net.W{l,ll}))
      net.X{l} = net.X{l} + net.W{l,ll} * net.Y{ll};
    end
  end
  net.Y{l} = nonlin(net.X{l}, net.layertypes{l-1}, net.nonlintrans{l});
end

output = net.Y{numel(net.layers)};
