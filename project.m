%% Exam project
% Double Pendulum Parameter Estimation
clear; clc;

% Known constants
g  = 9.81; % Gravity [m/s^2]
m1 = 100;  % Known mass 1
m2 = 60;   % Known mass 2

% True (unknown) parameters
L1_true = 2.0; % Length 1 [m]
L2_true = 1.5; % Length 2 [m]

% Time span
tspan = [0 10];
dt = 0.01;
t = tspan(1):dt:tspan(2);

% Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
theta0 = [0.2; 0; -0.1; 0];

[tsol, ysol] = ode45(@(t, y) double_pendulum_rhs(t, y, m1, m2, L1_true, L2_true, g), t, theta0);

% Interpolate to uniform time vector
theta1 = interp1(tsol, ysol(:,1), t);
theta2 = interp1(tsol, ysol(:,3), t);

% Noisy measurements
sigma_epsilon = 0.0001;
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
L1_est = params_est(1);
L2_est = params_est(2);

% Display results
fprintf('Estimation #1 - LSQ\n')
fprintf('True L1 = %.4f m, Estimated L1 = %.4f m\n', L1_true, L1_est);
fprintf('True L2 = %.4f m, Estimated L2 = %.4f m\n', L2_true, L2_est);

%% Estimation #2 - Gradient Descent
% Initial guess
params0 = [0, 1]';

% GD parameters
alpha = 1e-10; % Learning rate
max_iter = 1000;
tolerance = 1e-6;

for iter = 1:max_iter
    gradient = 2 * X' * (X*params0 - Y);
    params_new = params0 - alpha * gradient;
    
    if norm(params_new - params0) < tolerance
        break;
    end
    params0 = params_new;
end

L1_gd = params_new(1);
L2_gd = params_new(2);

% Display results
fprintf('Estimation #2 - Gradient Descent\n')
fprintf('True L1 = %.4f m, Estimated L1 = %.4f m\n', L1_true, L1_est);
fprintf('True L2 = %.4f m, Estimated L2 = %.4f m\n', L2_true, L2_est);

%% Estimation #3 - Instrumental Variable Estimation
% Using lagged noisy angles as instruments for regression matrix X

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

%% Functions

function dydt = double_pendulum_rhs(~, y, m1, m2, L1, L2, g)
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

