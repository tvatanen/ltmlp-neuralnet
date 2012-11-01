function fx = nonlin(x,type,trans,deriv)
% returns a nonlinear function of the input
% use deriv=1 for the derivative of the nonlinearity instead
% x may be a matrix

if (nargin<4)
  deriv = 0; % default
end;

if strcmp(type,'tanh')
  if deriv==0
    fx = bsxfun(@times,trans(:,3),tanh(x) + bsxfun(@plus, bsxfun(@times,trans(:,1),x), trans(:,2)));
  else
    fx = bsxfun(@times,trans(:,3),bsxfun(@plus, 1-tanh(x).^2, trans(:,1)));
  end
elseif strcmp(type,'rect')
  if (deriv==0),
    fx = max(0,x) + bsxfun(@plus, bsxfun(@times,trans(:,1),x), trans(:,2));
  else
    fx = bsxfun(@plus, 0.5+0.5*sign(x), trans(:,1));
  end
elseif (strcmp(type,'softmax'))
  if (deriv==0),
    fx = exp(bsxfun(@minus,x,max(x,[],1)));
    fx = bsxfun(@times, fx, 1./sum(fx,1));
  else
    fx = ones(size(x));
    % this is a trick which makes the backpropagation work correctly.
    % the derivative is incorrect, but elsewhere, a quadratic error is
    % assumed, which works fine with this kludge derivative.
  end;
elseif (strcmp(type,'linear'))
  if (deriv==0),
    fx = x;
  else
    fx = ones(size(x));
  end;
elseif (strcmp(type,'softsign'))
  if (deriv==0),
    fx = x ./ (1 + abs(x)) + bsxfun(@plus, bsxfun(@times,trans(:,1),x), trans(:,2));
  else
    fx = bsxfun(@plus, 1 ./ (1 + abs(x)).^2, trans(:,1));
  end;
elseif (strcmp(type,'tanh_noshort'))
  if (deriv==0),
    fx = tanh(x);
  else
    fx = 1-tanh(x).^2;
  end
else
  fx = NaN;
end