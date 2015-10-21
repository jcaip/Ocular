function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


A_1 = cat(2, ones(m,1),X); % computers A1 and adds 1's
A_2 = cat(2, ones(m,1), sigmoid(A_1*Theta1')); %computes A_2, adding bias term
h_theta = sigmoid(A_2*Theta2'); %computes hypothesis

plat_y = (y==1); %initialize large y matrix
for i=2:num_labels

	y_iter = (y==i);
	plat_y = [plat_y, y_iter]; %add each y_k to form the larger y matrix
end

J_individual = @(ys) sum((-plat_y(:,ys).*log(h_theta(:,ys))) - ((1-plat_y(:,ys)).*log(1-h_theta(:,ys))))/m;

J=sum(pararrayfun(3, J_individual, 1:num_labels));

Theta_1_reg = Theta1(:, 2:end);
Theta_2_reg = Theta2(:, 2:end);
J = J + (sum(sum(Theta_2_reg.^2))+sum(sum(Theta_1_reg.^2)))*(lambda)/(2*m); %computes regularized cost function
delta_3 = h_theta-plat_y;
delta_2 = delta_3*Theta2(:,2:end) .* sigmoidGradient(A_1*Theta1');

Theta2_grad = (delta_3'*A_2)/m;
Theta1_grad = (delta_2'*A_1)/m;

Theta1_grad(:,2:end)= Theta1_grad(:,2:end)+lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+lambda/m*Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
