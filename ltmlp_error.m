function [train_error, test_error] = ltmlp_error(net, train_input, train_output, test_input, test_output)

% Training error
current_train_output = ltmlp_ff(net, train_input);
if strcmp(net.options.task,'regression') || strcmp(net.options.task,'autoencoder')
  train_error = mean(sum((current_train_output-train_output).^2,1),2);
elseif strcmp(net.options.task,'classification')
  [~,maxI1] = max(current_train_output);
  [~,maxI2] = max(train_output);
  train_error = 100 - 100*sum(maxI1==maxI2)/length(maxI1);
else
  train_error = NaN;
end

% Test error
if nargin > 4
  current_test_output = ltmlp_ff(net, test_input);
  if strcmp(net.options.task,'regression') || strcmp(net.options.task,'autoencoder')
    test_error = mean(sum((current_test_output-test_output).^2,1),2);
  elseif strcmp(net.options.task,'classification')
    [~,maxI1]=max(current_test_output);
    [~,maxI2]=max(test_output);
    test_error = 100 - 100*sum(maxI1==maxI2)/length(maxI1);
  else
    test_error = NaN;
  end
end