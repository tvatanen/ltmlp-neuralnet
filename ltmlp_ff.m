function [output, net] = ltmlp_ff(net, input)

% this allows more args for output nonlinearity:
% if nargin > 2
%   exclude_output_nonlineariry = varargin{1};
% else
%   exclude_output_nonlineariry = 0;
% end

% Feedforward computations for ltmlp

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

% if ~exclude_output_nonlineariry && strcmp(net.layertypes{end},'softmax')
%   output_tmp = 
%   [~,maxI1]=max(net.Y{numel(net.layers)});
%   [~,maxI2]=max(valid_output);
%   valid_dur = 100 - 100*sum(maxI1==maxI2)/length(maxI1);
% end

output = net.Y{numel(net.layers)};
