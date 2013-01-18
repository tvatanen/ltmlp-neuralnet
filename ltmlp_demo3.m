%% This demo replicates results in [1], Section 5, MNIST classification

% load zero-mean MNIST data
load('data/mnist_preprocessedd.mat');

% Network size 784-800-800-10
nl = [800 800];
nltypes = {'tanh', 'tanh', 'softmax'};

%% 1

% random number generator seed
rng(9291);

options = ltmlp_opt('stepsize', 0.2, 'runtime', 150, 'weightdecay', 0., ...
  'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
  'numtransf', 3, 'batchsize', 1000, 'mominit', 0.5, 'momentum', 0.9, 'burnin', 50, ...
  'ratedecay', 0.9);
net = ltmlp(nl, nltypes, 784, 10);
net = ltmlp_init(net, options, data_input);
[net, res1] = ltmlp_train2(net, data_input, data_output, valid_input, valid_output);

%res1_4 : T=50, stepsize=0.2
%% 2

% rng(9291);
% options = ltmlp_opt('stepsize', 0.7, 'runtime', 200, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 2, 'updatetransf', 1000, 'batchsize', 1000, 'momentum', 0.5);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init2(net, options, data_input);
% [~, res2] = ltmlp_train_constant2(net, data_input, data_output, valid_input, valid_output, 0.9);
% 
% save('res2_2', 'res2')
%% 3

% rng(9291);
% options = ltmlp_opt('stepsize', 0.05, 'runtime', 500, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 0, 'updatetransf', 1000, 'batchsize', 1000, 'momentum', 0.5);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init2(net, options, data_input);
% [~, res3] = ltmlp_train_constant2(net, data_input, data_output, valid_input, valid_output, 0.9, 50);
% 
% save('res3', 'res3')
%%
% 
% nl = [400 400 400];
% nltypes = {'tanh', 'tanh', 'tanh', 'softmax'};
% 
% rng(9291);
% options = ltmlp_opt('stepsize', 0.3, 'runtime', 200, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 3, 'updatetransf', 1000, 'batchsize', 1000, 'momentum', 0.5);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init2(net, options, data_input);
% [net, res1_big] = ltmlp_train_constant2(net, data_input, data_output, valid_input, valid_output, 0.9, 50);
% 
% [~, net] = ltmlp_ff(net, data_input);         
% outputs = net.Y;
% outputvar1_big = var(outputs{2}');
% outputvar2_big = var(outputs{3}');
% outputvar3_big = var(outputs{4}');
% save('outputvars_big', 'outputvar1_big', 'outputvar2_big', 'outputvar3_big');
% 
% save('res1_big2', 'res1_big')

% big1: 0.4, 20
% big2: 0.3, 50
%%
% nl = [400 400 400];
% nltypes = {'tanh', 'tanh', 'tanh', 'softmax'};
% 
% rng(9291);
% options = ltmlp_opt('stepsize', 0.7, 'runtime', 151, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 2, 'updatetransf', 1000, 'batchsize', 1000, 'momentum', 0.5);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init2(net, options, data_input);
% [net, res2_big] = ltmlp_train_constant2(net, data_input, data_output, valid_input, valid_output, 0.9, 50);
% 
% [~, net] = ltmlp_ff(net, data_input);         
% outputs = net.Y;
% outputvar1_big = var(outputs{2}');
% outputvar2_big = var(outputs{3}');
% outputvar3_big = var(outputs{4}');
% save('outputvars2_big', 'outputvar1_big', 'outputvar2_big', 'outputvar3_big');
% 
% save('res2_big', 'res2_big')

%%
% rng(9291);
% options = ltmlp_opt('stepsize', 0.05, 'runtime', 200, 'weightdecay', 0., ...
%   'inputnoise', 0.3, 'task', 'classification', 'verbose', 1, ...
%   'numtransf', 0, 'updatetransf', 1000, 'batchsize', 1000, 'momentum', 0.5);
% net = ltmlp(nl, nltypes, 784, 10);
% net = ltmlp_init.2(net, options, data_input);
% [~, res3_big] = ltmlp_train_constant2(net, data_input, data_output, valid_input, valid_output, 0.9, 50);
% 
% save('res3_big', 'res3_big')