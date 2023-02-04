% estimate x (mean and covariance)

%load dataset
%load dataset1.mat;

% % Required matrices
% k = size(v);
% 
% % assuming initial measurement as initial state
% x0_check = l - r(1);
% 
% % the uncertainty of initial state will be the same as uncertainty in
% % initial measurement
% P0_check = r_var;
% 
% % measurements
% y = l - r;
% 
% % z matrix = [v, y].T
% v = v * 0.1;
% v(1) = x0_check; 
% z = [v; y]; 
% 
% % A matrix in lifted form for given motion model
% A = ones(k(1), k(1));
% A = tril(A);
% 
% % A inverse
% A_inv = sparse(inv(A));
% 
% % C matrix
% C = eye(k(1));
% 
% % H = [A_inv, C].T
% H = [A_inv; C];
% 
% % Q process noise
% process_noise = v_var*ones(1, k(1));
% process_noise(1) = P0_check;
% Q = diag(process_noise);
% 
% % R sensor noise
% R = r_var*diag(ones(1, k(1)));
% 
% % W = [Q, 0; 0 R] blockdiagonal matrix
% W = blkdiag(Q, R);
% W_inv = sparse(inv(W));
% 
% % H.T*W_inv*H
% lhs = transpose(H) * W_inv * H;
% spy(lhs);

% initalize
prompt = 'What is the delta ? ';
delta = input(prompt);
Phat_f = ones(k(1), 1);
xhat_f = ones(k(1), 1);
Pcheck_f = ones(k(1), 1);
xcheck_f = ones(k(1), 1);
xcheck_0f = y(1);
Pcheck_0f = Q(1, 1);
Pcheck_f(1) = Pcheck_0f;
xcheck_f(1) = xcheck_0f;
K = Pcheck_0f * transpose(C(1, 1)) * inv((C(1, 1) * Pcheck_0f * transpose(C(1, 1)) + R(1, 1))); % P0_check = Q(1, 1)
Phat_f(1) = (1 - K * C(1, 1)) * Pcheck_0f;
xhat_f(1) = xcheck_0f + K * (y(1) - C(1, 1) * xcheck_0f);


for i = 2 : k(1) %matlab indexing starts from 1
    % forward pass
    % prediction
    xcheck_f(i) = A(i-1, i-1) * xhat_f(i - 1) + v(i);
    Pcheck_f(i) = A(i - 1, i - 1) * Phat_f(i-1) * transpose(A(i - 1, i - 1)) + Q(i, i);
    % correction
    if  mod(i, delta) == 0
        K = Pcheck_f(i) * transpose(C(i, i)) * inv((C(i, i) * Pcheck_f(i) * transpose(C(i, i)) + R(i, i)));
        Phat_f(i) = (1 - K*C(i, i)) * Pcheck_f(i);
        xhat_f(i) = xcheck_f(i) + K * (y(i) - C(i, i) * xcheck_f(i)); 
    else
        Phat_f(i) = Pcheck_f(i);
        xhat_f(i) = xcheck_f(i);
    end
end
% initialize backward pass
xhat = ones(k(1), 1);
Phat = ones(k(1), 1);
Phat(k(1)) = Phat_f(k(1));
xhat(k(1)) = xhat_f(k(1));
for i = k(1) : -1 : 2
    % backward pass
    xhat(i - 1) = xhat_f(i - 1) + Phat_f(i - 1) * transpose(A(i - 1, i - 1)) * inv(Pcheck_f(i - 1)) * (xhat(i) - xcheck_f(i));
    Phat(i - 1) = Phat_f(i - 1) + (Phat_f(i - 1) * transpose(A(i - 1, i - 1)) * inv(Pcheck_f(i))) * (Phat(i) - Pcheck_f(i)) * transpose(Phat_f(i - 1) * transpose(A(i - 1, i - 1)) * inv(Pcheck_f(i)));
end

% error plots

error = x_true - xhat;
x_axis = 1 : k(1);
y_axis = error;
figure(1);
plot(x_axis, y_axis);
hold on;
% 3 sigma plots
plot(sqrt(Phat) * 3, '--')
hold on;
plot(- sqrt(Phat) * 3, '--')
hold on;
title('Error Plots');
xlabel('Timesteps(s)');
ylabel('Error(m)');
% histogram
figure(2);
histogram(error);
title(sprintf('Mean - %f, Std Deviation - %f', mean(error), std(error)), '$\delta - 1000$', 'Interpreter', 'latex');
xlabel('Error (m)');