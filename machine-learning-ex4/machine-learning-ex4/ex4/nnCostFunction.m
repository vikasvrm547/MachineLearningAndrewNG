function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

temp_j = 0;
for i=1:m,
  for k=1:num_labels,
    temp_y = 0;
     if y(i,1) == k,
       temp_y = 1;
     end
    
    J = J + (temp_y * log(h2(i,k)) + (1 - temp_y) * log(1 - h2(i,k)));
  end
end

J = (-1/m) *J;

%regularize = (lambda/(2*m))*(sum((Theta1(:,2: length(Theta1)).^2)(:)) + sum((Theta2(:,2: length(Theta2)).^2)(:)))
regularize1 = 0;

for j = 1:hidden_layer_size,
  for k=2: input_layer_size+1,
    regularize1 = regularize1 + Theta1(j,k)^2;
  end
end

regularize2 = 0;

for j = 1:num_labels,
  for k=2: hidden_layer_size+1,
    regularize2 = regularize2 + Theta2(j,k)^2;
  end
end

regularize =  (lambda/(2*m)) * (regularize1 + regularize2);
J = J + regularize;

% compute gradient 

  a1= [ones(m, 1) X];
  z2 = a1* Theta1';
  a2 = [ones(m, 1) sigmoid(z2)];
  
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  y_matrix = zeros(m,num_labels);
  
  for i=1:num_labels,
    y_matrix(:,i) = y==i;  
  end
  
  delta3 = a3 - y_matrix;
  
  Theta2_grad = delta3' * a2;
  
  delta2 = (delta3*Theta2(:,2:size(Theta2,2))).*sigmoidGradient(z2);
  
  Theta1_grad = delta2' * a1;

%end
% -------------------------------------------------------------
Theta1_grad_wr = (1/m)*Theta1_grad(:,1);
Theta1_grad_r = (1/m)*Theta1_grad(:,2: size(Theta1_grad,2))  +  (lambda/m)*Theta1(:,2: size(Theta1,2));
Theta2_grad_wr = (1/m)*Theta2_grad(:,1);
Theta2_grad_r = (1/m)*Theta2_grad(:,2: size(Theta2_grad,2))  +  (lambda/m)*Theta2(:,2: size(Theta2,2));;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad_wr(:) ; Theta1_grad_r(:); Theta2_grad_wr(:); Theta2_grad_r(:)];


end
