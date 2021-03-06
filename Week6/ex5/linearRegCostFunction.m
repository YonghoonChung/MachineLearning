function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%%%%%% 1.2 %%%%%%%%%%
h_theta = X*theta; % 12 X 1 * 1 X 1 -> 12 X 1
J = (1/(2*m))*sum((h_theta-y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

%%%%%%% 1.3 %%%%%

% regularizedGrad = (lambda/m)*theta(2:end);
% grad(1) = (1/m)*sum(h_theta-y)'*X(:,1);
% grad(2:end) = (1/m)*sum(h_theta-y)*X' + regularizedGrad;

% dimension
%     X       : 12 X 2
%     y       : 12 X 1
%     theta   : 2 X 1
%     grad    : 2 X 1
%     h_theta : 12 X 2 * 2 X 1 -> 12 X 1

% grad = (1/m)*(h_theta-y)'*X + (lambda/m)*theta;
grad(1) = (1/m)*(X(:,1)'*(h_theta-y));
grad(2:end) = (1/m)*(X(:,2:end)'*(h_theta-y))+ ((lambda/m)*theta(2:end));








% =========================================================================

grad = grad(:);

end
