% remember to add ltmlp to your path
addpath('ltmlp')

%% Some options explained below:
%
% 'runtime' is the runtime in seconds.
%
% 'stepsize' for the first half of the training. After that the network
% is "annealed", i.e., the stepsize will decay to zero.
% 
% 'task' can so far obtain three values: 'regression', 'classification' and
% 'autoencoder'. The 'task' is used to select the way error is computed.
%
% 'errorevals' determines how many times training and test errors
% are evaluated during the training
%
% 'verbose' determines verbosity level. Defaults to 1
%   0: quiet
%   1: some printing
%   2: some plotting (might be buggy)
%   4: debugging
%
% 'numtransf' determines the number of transformation parameters (alpha,
% beta, gamma). Defaults to 2, i.e., no gamma is used.
%
% 'fixedtransf' option can be used to use fixed transformations throughout 
% whole training, e.g., "'fixedtransf', [-0.5 0 6]" will use fixed 
% aplha = -0.5, beta = 0, gamma = 6
%
% 'updatetrans' determines how often (in iterations) transformations are
% updated.

% number of neurons in hidden layers (also determines the number of hidden
% layers)
nl = [200 200];

% types of nonlinearities in hidden layers and output layer
nltypes = {'tanh', 'tanh', 'softmax'};

% create the ltmlp object
% usage: ltmlp(hiddenlayers, layertypes, input_dim, output_dim, [varargin])
net = ltmlp(nl, nltypes, 200, 10);

% load following variables:
% data_input  data_mean  data_output  pca_W  scale  valid_input
% valid_output
load(sprintf('mnist/mnist_preprocessed_%d.mat',200));

data_input = single(data_input);
valid_input = single(valid_input);

%% Example configurations:
% Results are returned in res struct
% network without gamma
options = ltmlp_opt('stepsize', 0.3, 'runtime', 15*60, 'weightdecay', 0.0001, ...
  'inputnoise', 0.4, 'task', 'classification', 'errorevals', 100, 'verbose', 1, ...
  'numtransf', 2, 'updatetransf', 100);
% initialize ...
net = ltmlp_init(net, options, data_input);
% ... and train
[net res] = ltmlp_train(net, data_input, data_output, valid_input, valid_output);  

% network with fixed transformations: alpha = -0.5, beta = 0, gamma = 1 (no gamma)
options = ltmlp_opt('stepsize', 0.3, 'runtime', 15*60, 'weightdecay', 0.0001, ...
  'inputnoise', 0.4, 'task', 'classification', 'errorevals', 100, 'verbose', 1, ...
  'numtransf', 3, 'updatetransf', 100, 'fixedtransf', [-0.5 0 1]);
net = ltmlp_init(net, options, data_input);
[net res] = ltmlp_train(net, data_input, data_output, valid_input, valid_output);  

% network with three tranformations (alpha,beta,gamma) and smaller stepsize
options = ltmlp_opt('stepsize', 0.06, 'runtime', 15*60, 'weightdecay', 0.0001, ...
  'inputnoise', 0.4, 'task', 'classification', 'errorevals', 100, 'verbose', 1, ...
  'numtransf', 3, 'updatetransf', 100);
net = ltmlp_init(net, options, data_input);
[net res] = ltmlp_train(net, data_input, data_output, valid_input, valid_output);  

% network with fixed transformations: alpha = -0.5, beta = 0, gamma = 5 and
% smaller stepsize
options = ltmlp_opt('stepsize', 0.06, 'runtime', 15*60, 'weightdecay', 0.0001, ...
  'inputnoise', 0.4, 'task', 'classification', 'errorevals', 100, 'verbose', 1, ...
  'numtransf', 3, 'updatetransf', 100, 'fixedtransf', [-0.5 0 5]);
net = ltmlp_init(net, options, data_input);
[net res] = ltmlp_train(net, data_input, data_output, valid_input, valid_output);

