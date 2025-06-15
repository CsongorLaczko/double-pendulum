%% Exam project
% 17. Logging timber by helicopter
clear; clc;

% Known constants
g  = 9.81; % Gravity [m/s^2]
m1 = 100;  % Known mass 1 [kg]
m2 = 60;   % Known mass 2 [kg]

% True (unknown) parameters
L1_true = 2.0; % Length 1 [m]
L2_true = 1.5; % Length 2 [m]

% Time span
tspan = [0 10];
dt = 0.01;
t = tspan(1):dt:tspan(2);

% Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
theta0 = [0.2; 0; -0.1; 0];

[tsol, ysol] = ode45(@(t, y) double_pendulum_rhs(y, m1, m2, L1_true, L2_true, g), t, theta0);

% Interpolate to uniform time vector
theta1 = interp1(tsol, ysol(:,1), t);
theta2 = interp1(tsol, ysol(:,3), t);

% Noisy measurements
sigma_epsilon = 0.00001;
theta1_noisy = theta1 + sigma_epsilon * randn(size(theta1));
theta2_noisy = theta2 + sigma_epsilon * randn(size(theta2));

% Numerical second derivatives using finite differences
ddtheta1_noisy = gradient(gradient(theta1_noisy, dt), dt);
ddtheta2_noisy = gradient(gradient(theta2_noisy, dt), dt);

% Number of samples
N = length(t);

%% Estimation #1 - LSQ

% Prepare regression matrices
X = zeros(2*N, 2);
Y = zeros(2*N, 1);

for i = 1:N
    a = (m1 + m2) * ddtheta1_noisy(i);
    b = m2 * ddtheta2_noisy(i);
    c = (m1 + m2) * g * theta1_noisy(i);

    d = m2 * ddtheta1_noisy(i);
    e = m2 * ddtheta2_noisy(i);
    f = m2 * g * theta2_noisy(i);

    % First equation (divided by L1)
    X(2*i-1, :) = [a, b];
    Y(2*i-1) = -c;

    % Second equation (divided by L2)
    X(2*i, :) = [d, e];
    Y(2*i) = -f;
end

% Least squares solution
params_est = (X' * X) \ (X' * Y);
L1_lsq = params_est(1);
L2_lsq = params_est(2);

% Display results
fprintf('Estimation #1 - LSQ\n')
fprintf('True L1 = %.4f m, Estimated L1 = %.4f m\n', L1_true, L1_lsq);
fprintf('True L2 = %.4f m, Estimated L2 = %.4f m\n', L2_true, L2_lsq);

%% Estimation #2 - Gradient Descent
% Initial guess
params_gd = [1, 1]';

% GD parameters
alpha = 0.01; % Learning rate
max_iter = 5000;
tolerance = 1e-4;

fprintf('Estimation #2 - Gradient Descent\n')

for iter = 1:max_iter
    loss_current = compute_loss_gd(params_gd, t, theta0, m1, m2, g, theta1_noisy, theta2_noisy);

    % Numerical gradient (finite differences)
    grad = zeros(2,1);
    h = 1e-6;
    for i = 1:2
        params_up = params_gd;
        params_down = params_gd;
        params_up(i) = params_up(i) + h;
        params_down(i) = params_down(i) - h;

        loss_up = compute_loss_gd(params_up, t, theta0, m1, m2, g, theta1_noisy, theta2_noisy);
        loss_down = compute_loss_gd(params_down, t, theta0, m1, m2, g, theta1_noisy, theta2_noisy);

        grad(i) = (loss_up - loss_down) / (2*h);
    end

    % Update parameters
    params_new = params_gd - alpha * grad;

    % Check convergence
    if norm(params_new - params_gd) < tolerance
        fprintf('Converged at iteration %d\n', iter);
        break;
    end

    params_gd = params_new;

    if mod(iter, 100) == 0
        fprintf('Iter %d, Loss %.4e, L1=%.4f, L2=%.4f\n', iter, loss_current, params_gd(1), params_gd(2));
    end
end

L1_gd = params_gd(1);
L2_gd = params_gd(2);

% Display results
fprintf('True L1 = %.4f m, Estimated L1 = %.4f m\n', L1_true, L1_gd);
fprintf('True L2 = %.4f m, Estimated L2 = %.4f m\n', L2_true, L2_gd);

%% Estimation #3 - Instrumental Variable Estimation
% Construct instruments matrix Z from lagged noisy measurements:
% Shift theta1_noisy and theta2_noisy by one step to serve as instruments
Z = zeros(2*N, 2);

for i = 2:N
    Z(2*i-1, :) = [theta1_noisy(i-1), theta2_noisy(i-1)];
    Z(2*i, :) = [theta1_noisy(i-1), theta2_noisy(i-1)];
end

% For the first sample, repeat first instruments
Z(1, :) = Z(3, :);
Z(2, :) = Z(4, :);

% Compute IV estimate
params_iv = (Z' * X) \ (Z' * Y);

L1_iv = params_iv(1);
L2_iv = params_iv(2);

fprintf('Estimation #3 - Instrumental Variable Estimation\n')
fprintf('True L1 = %.4f m, Estimated L1 = %.4f m\n', L1_true, L1_iv);
fprintf('True L2 = %.4f m, Estimated L2 = %.4f m\n', L2_true, L2_iv);

%% Plot
% Simulate with LSQ estimates
[tsol_LSQ, ysol_LSQ] = ode45(@(t,y) double_pendulum_rhs(y,m1,m2,L1_lsq,L2_lsq,g), t, theta0);
theta1_est_LSQ = interp1(tsol_LSQ, ysol_LSQ(:,1), t);
theta2_est_LSQ = interp1(tsol_LSQ, ysol_LSQ(:,3), t);

% Simulate with GD estimates
[tsol_GD, ysol_GD] = ode45(@(t,y) double_pendulum_rhs(y,m1,m2,L1_gd,L2_gd,g), t, theta0);
theta1_est_GD = interp1(tsol_GD, ysol_GD(:,1), t);
theta2_est_GD = interp1(tsol_GD, ysol_GD(:,3), t);

% Simulate with IV estimates
[tsol_IV, ysol_IV] = ode45(@(t,y) double_pendulum_rhs(y,m1,m2,L1_iv,L2_iv,g), t, theta0);
theta1_est_IV = interp1(tsol_IV, ysol_IV(:,1), t);
theta2_est_IV = interp1(tsol_IV, ysol_IV(:,3), t);

% Plot
figure;
subplot(2,1,1);
plot(t, theta1, 'b', 'LineWidth', 1.5); hold on;
plot(t, theta1_est_LSQ, 'r--', 'LineWidth', 1.2);
plot(t, theta1_est_GD, 'g--', 'LineWidth', 1.2);
plot(t, theta1_est_IV, 'm--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\theta_1 [rad]');
title('Angle \theta_1 and Estimates');
legend('True', 'LSQ Estimate', 'GD Estimate', 'IV Estimate');
grid on;

subplot(2,1,2);
plot(t, theta2, 'b', 'LineWidth', 1.5); hold on;
plot(t, theta2_est_LSQ, 'r--', 'LineWidth', 1.2);
plot(t, theta2_est_GD, 'g--', 'LineWidth', 1.2);
plot(t, theta2_est_IV, 'm--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('\theta_2 [rad]');
title('Angle \theta_2 and Estimates');
legend('True', 'LSQ Estimate', 'GD Estimate', 'IV Estimate');
grid on;

%% Functions

function dydt = double_pendulum_rhs(y, m1, m2, L1, L2, g)
    theta1 = y(1);
    dtheta1 = y(2);
    theta2 = y(3);
    dtheta2 = y(4);

    % System matrix
    A = [(m1 + m2)*L1^2,  m2*L1*L2;
         m2*L1*L2,        m2*L2^2];

    % RHS vector
    b = -[(m1 + m2)*g*L1*theta1;
          m2*g*L2*theta2];

    ddtheta = A \ b;

    dydt = zeros(4,1);
    dydt(1) = dtheta1;
    dydt(2) = ddtheta(1);
    dydt(3) = dtheta2;
    dydt(4) = ddtheta(2);
end

function loss = compute_loss_gd(params, t, theta0, m1, m2, g, theta1_noisy, theta2_noisy)
    [tsim, ysim] = ode45(@(t,y) double_pendulum_rhs(y,m1,m2,params(1),params(2),g), t, theta0);

    theta1_pred = interp1(tsim, ysim(:,1), t);
    theta2_pred = interp1(tsim, ysim(:,3), t);

    loss = mean((theta1_pred - theta1_noisy).^2 + (theta2_pred - theta2_noisy).^2);
end

