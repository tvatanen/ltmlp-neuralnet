function opt = ltmlp_opt(varargin)


% check if options struct is provided
if isstruct(varargin{1}) && isfield(varargin{1},'type') && ...
    strcmp(varargin{1}.type, 'ltmlp options')
  opt = varargin{1};

else % otherwise use defaults
  
  % shortcuts 1=no, 2=yes, 3=yes with transformations, 4=yes with transf and
  opt.type = 'ltmlp options';
  opt.task = 'regression';         % task for the ltmlp
                                   % regression / classification / 
                                   % autoencoder
  opt.num_transf = 2;              % number of transformations
                                   % (alpha/beta/gamma)
  opt.bias_init_scale = 0.5;       %
  opt.ratedecay = 0.9;             % learning rate decay term
  opt.momentum = 0.9;              % momentum for the gradient
  opt.mominit = 0.5;               % initial momentum
  opt.burnin = 50;                 % burn-in time (epochs, momentum increases)
  opt.minibatchsize = 1000;        % mini-batch size
  opt.weight_decay = 0;            % weight decay term, lambda
  opt.transf_every_n_iters = 1;    % how often to update transformations
  opt.n_error_evals = 100;         % how many times to evaluate error(s)
  opt.stepsize = 1.0;              % learning rate in the beginning 
  opt.runtime = 60000;             % runtime in epochs
  opt.input_noise = 0.0;           % add noise to input activations
  opt.gammatype = 1;               % determines how gamma is computed
  opt.verbose = 1;                 % level of verbosity
                                   %   (0) no output
                                   %   (1) print info (default)
                                   %   (2) includes plotting
  opt.fixed_transf = 0;            % fixed transformation parameters

end

% process varargin
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch lower(varargin{i}),
      % argument IDs
      case 'stepsize', i=i+1; opt.stepsize = varargin{i};
      case 'runtime', i=i+1; opt.runtime = varargin{i};
      case 'weightdecay', i=i+1; opt.weight_decay = varargin{i};
      case 'inputnoise', i=i+1; opt.input_noise = varargin{i};
      case 'task', i=i+1; opt.task = varargin{i};
      case 'errorevals', i=i+1; opt.n_error_evals = varargin{i};
      case 'verbose', i=i+1; opt.verbose = varargin{i};
      case 'fixedtransf', i=i+1; opt.fixed_transf = varargin{i};
      case 'numtransf', i=i+1; opt.num_transf = varargin{i};
      case 'updatetransf', i=i+1; opt.transf_every_n_iters = varargin{i};
      case 'batchsize', i=i+1; opt.minibatchsize = varargin{i};
      case 'gammatype', i=i+1; opt.gammatype = varargin{i};
      case 'momentum', i=i+1; opt.momentum = varargin{i};
      case 'mominit', i=i+1; opt.mominit = varargin{i};        
      case 'burnin', i=i+1; opt.burnin = varargin{i};
      case 'ratedecay', i=i+1; opt.ratedecay = varargin{i};
     otherwise; argok=0; 
    end
  else
    argok = 0; 
  end
  if ~argok
    disp(['(ltmlp_opt) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1;
end