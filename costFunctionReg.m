function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z=X*theta;
e=exp(-z);
denominator=e.+1;
g1=1./denominator;
g2=1.-g1;
n=size(theta,1);
thetaspecial=theta(2:n,1);
power=thetaspecial'*thetaspecial;

J=(1.0/m).*(-y'*log(g1)-(1.-y')*log(g2))+(lambda*power)/(2*m);

gradtotal=(1.0/m).*X'*(g1.-y);

grad0=gradtotal(1,1);

gradspecial=gradtotal(2:n,1).+(lambda/m).*theta(2:n,1);

grad=[gradtotal(1,1); gradspecial];






% =============================================================

end
