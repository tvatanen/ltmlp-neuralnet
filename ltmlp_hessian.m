function Hessian = ltmlp_hessian(net, input, output, gradient)

num_samples = size(output, 2);
nlayers = numel(net.layers);
gammareco = cell(1,nlayers);
gradbias = cell(nlayers,1);
nonlintypes = net.layertypes;
gradW = cell(nlayers,nlayers-1);
opt = net.options;

% epsilon to be added each element of W
h = 1e-3;

Hessian = zeros(net.num_params);
gradW_old = zeros(net.num_params,1);
gradW_new = zeros(net.num_params,2);

% concatenate old gradient into a vector
w_count = 1;
for l = 2:nlayers
  for ll=1:l-1,
    if ~isempty(net.W{l,ll})
      gradW_tmp = gradient{l,ll}(:);
      gradW_old(w_count:w_count + length(gradW_tmp) - 1) = gradW_tmp;
      w_count = w_count + length(gradW_tmp);
    end
  end
end

hessian_row = 1;
for l = 2:nlayers
  for ll=1:l-1,
    if ~isempty(net.W{l,ll})
      % Alternate one element of net.W{l,ll} at the time
      % and compute one row of hessian
      for i = 1:numel(net.W{l,ll})
        Wi_old = net.W{l,ll}(i);        

        for sgn = 1:2
          
          if sgn == 1
            net.W{l,ll}(i) = net.W{l,ll}(i) + h;
          elseif sgn == 2
            net.W{l,ll}(i) = net.W{l,ll}(i) - h;
          end
          
          % Feed-forward
          [updated_output, net] = ltmlp_ff(net, input);

          % Compute reconstruction error
          reco_err = updated_output - output;

          % Backpropagate
          gammareco{nlayers} = nonlin(net.X{nlayers}, nonlintypes{nlayers-1}, net.nonlintrans{nlayers}, 1) .* reco_err;
          for k = nlayers-1:-1:2,
            gammareco{k} = 0;
            for kk=(k+1):nlayers,
              if ~isempty(net.W {kk,k})
                gammareco{k} = gammareco{k} + (net.W{kk,k}' * gammareco{kk}) .* ...
                  nonlin(net.X{k}, nonlintypes{k-1}, net.nonlintrans{k}, 1);
              end
            end
          end

          % Gradient computations
          for k = 2:nlayers
            % Regular gradient
            gradbias{k} = sum(gammareco{k}, 2) / num_samples;
            for kk=1:k-1,
              if ~isempty(net.W{k,kk})
                gradW{k,kk} = gammareco{k} * net.Y{kk}' / num_samples;
              end
            end
            if opt.weight_decay>0
              % Weight decay
              gradbias{k} = gradbias{k} + opt.weight_decay * net.bias{k};
              for kk=1:k-1,
                if ~isempty(net.W{k,kk})
                  gradW{k,kk} = gradW{k,kk} + opt.weight_decay * net.W{k,kk};
                end
              end
            end
          end

          % Concatenate a new gradient after altering the parameter number w_count
          w_count = 1;
          for k = 2:nlayers
            for kk=1:k-1,
              if ~isempty(net.W{k,kk})
                gradW_tmp = gradW{k,kk}(:);
                gradW_new(w_count:w_count + length(gradW_tmp) - 1,sgn) = gradW_tmp;
                w_count = w_count + length(gradW_tmp);
              end
            end
          end
          net.W{l,ll}(i) = Wi_old;          
        end

        Hessian(hessian_row, :) = (gradW_new(:,1) - gradW_new(:,2)) / (2*h);
        
        hessian_row = hessian_row + 1;
      end
    end
  end
end

%keyboard