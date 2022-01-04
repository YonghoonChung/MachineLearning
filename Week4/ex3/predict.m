function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%{The matrix X contains the examples in rows. 
% When you complete the code in predict.m, you will need to add the column of 1's to the matrix. 
% The matrices Theta1 and Theta2 contain the parameters for each unit in rows. 
% Specically, the first row of Theta1 corresponds to the first hidden unit in the second layer. 
% In MATLAB, when you compute , be sure that you index (and if necessary, transpose) X correctly so that you get  as a column vector.
%}

a1 = [ones(size(X,1),1) X];
z2 = a1* Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(a2 * Theta2');
[prob,p] = max(a3, [],2);



% =========================================================================


end
