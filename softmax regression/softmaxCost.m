function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% Compute the hypothesis matrix
H = theta * data;
H = bsxfun(@minus, H, max(H, [], 1));
H = exp(H);
H = bsxfun(@rdivide, H, sum(H));

% Cost value for cost function J.
% cost = -1 / numCases * groundTruth(:)' * log(H(:)) + lambda / 2 * sum(theta(:).^2);
cost = -1 / numCases * sum(sum(groundTruth .* log(H))) + 0.5 * lambda * sum(theta(:).^2) ;

% partial gradient for theta
thetagrad = -1 / numCases * (groundTruth - H) * data' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);

end

