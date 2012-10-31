function errstring = ltmlp_consist(net, type, input_dim, output_dim)
%CONSIST Check that arguments are consistent.
%
%	Description
%
%	ERRSTRING = CONSIST(NET, TYPE, INPUTS) takes a network data structure
%	NET together with a string TYPE containing the correct network type,
%	a matrix INPUTS of input vectors and checks that the data structure
%	is consistent with the other arguments.  An empty string is returned
%	if there is no error, otherwise the string contains the relevant
%	error message.  If the TYPE string is empty, then any type of network
%	is allowed.
%
%	ERRSTRING = CONSIST(NET, TYPE) takes a network data structure NET
%	together with a string TYPE containing the correct  network type, and
%	checks that the two types match.
%
%	ERRSTRING = CONSIST(NET, TYPE, INPUTS, OUTPUTS) also checks that the
%	network has the correct number of outputs, and that the number of
%	patterns in the INPUTS and OUTPUTS is the same.  The fields in NET
%	that are used are
%	  type
%	  nin
%	  nout
%
%	See also
%	MLPFWD
%

%	Copyright (c) Ian T Nabney (1996-2001)

% Assume that all is OK as default
errstring = '';

% If type string is not empty
if ~isempty(type)
  % First check that net has type field
  if ~isfield(net, 'type')
    errstring = 'Data structure does not contain type field';
    return
  end
  % Check that net has the correct type
  s = net.type;
  if ~strcmp(s, type)
    errstring = sprintf('Model type %s does not match expected type %s', ...
      s, type);
    return
  end
end

% If input_dim is present, check that it is correct
if nargin > 2
  if ~isfield(net, 'layers')
    errstring = 'Data structure does not contain layers field';
    return
  end

  if net.layers(1) ~= input_dim
    errstring = sprintf('Dimension of inputs %s does not match the input layer of network', ...
      num2str(input_dim));
    return
  end
end

% If output_dim present, check that it is correct
if nargin > 3
  if net.layers(end) ~= output_dim
    errstring = sprintf('Dimension of outputs %s does not match the output layer of network', ...
      num2str(output_dim));
    return
  end
end
