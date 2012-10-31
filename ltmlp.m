function net = ltmlp(hiddenlayers, layertypes, input_dim, output_dim, varargin)

net.type = 'ltmlp network';
net.layers = [input_dim hiddenlayers output_dim];
nlayers = numel(net.layers);
net.layertypes = layertypes;

% W contains weights { to which layer, from which layer }
net.W = cell(nlayers,nlayers-1); 
net.bias = cell(nlayers,1);          % bias term
%net.Y = cell(nlayers,1);
%net.X = cell(nlayers,1);