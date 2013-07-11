%% This demo replicates results in [1], Section 5, MNIST classification

% load zero-mean MNIST data
load('data/mnist_preprocessed.mat');

% Network size 784-800-800-10
nl = [800 800];
nltypes = {'tanh', 'tanh', 'softmax'};

%% Network with all three transformations

% random number generator seed
rng(9291);
options = ltmlp_opt('stepsize', 0.2, 'runtime', 150, 'weightdecay', 0., ...
  'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
  'numtransf', 3, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
  'ratedecay', 0.9);
net = ltmlp(nl, nltypes, 784, 10);
net = ltmlp_init(net, options, data_input);
[~, res1] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);

%% Network without gamma (only two transformations)

rng(9291);
options = ltmlp_opt('stepsize', 0.7, 'runtime', 150, 'weightdecay', 0., ...
  'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
  'numtransf', 2, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
  'ratedecay', 0.9);
net = ltmlp(nl, nltypes, 784, 10);
net = ltmlp_init(net, options, data_input);
[~, res2] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);

%% Network without transformations

rng(9291);
options = ltmlp_opt('stepsize', 0.05, 'runtime', 150, 'weightdecay', 0., ...
  'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
  'numtransf', 0, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
  'ratedecay', 0.9);

net = ltmlp(nl, nltypes, 784, 10);
net = ltmlp_init(net, options, data_input);
[~, res3] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);

%% Three-layer networks
% 
% nl = [400 400 400];
% nltypes = {'tanh', 'tanh', 'tanh', 'softmax'};
% 
% rng(9291);
% options = ltmlp_opt('stepsize', 0.3, 'runtime', 150, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 3, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
%   'ratedecay', 0.9);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init(net, options, data_input);
% [~, res1_big] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);
% 
% 
% rng(9291);
% options = ltmlp_opt('stepsize', 0.7, 'runtime', 150, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 2, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
%   'ratedecay', 0.9);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init(net, options, data_input);
% [~, res2_big] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);
% 
% rng(9291);
% options = ltmlp_opt('stepsize', 0.05, 'runtime', 150, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 0, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
%   'ratedecay', 0.9);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init(net, options, data_input);
% [~, res3_big] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);
% 
% save('comfirm_results', 'res1', 'res2', 'res3', 'res1_big', 'res2_big', 'res3_big')