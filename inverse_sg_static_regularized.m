% this script test static inverse SG with regularized follower.
% supervised learning, batch training, 1000 GD steps instead of solving opt.

close all
clearvars
rng(123)

m = 4;
n = 3;
U = rand(m, n);
V = rand(m, n);
V0 = rand(m,n);     % initial estimate of V
lam = 10;
N = 100;

% sample N data points
[Xdata, Ydata, Ymix] = sample_strategy(V, lam, N);

% perform GD by min cross entropy -y*log(exp/sum(exp))
ITER_MAX = 10000;
s = 1e-2;
Vk = V0;
for i = 1: ITER_MAX
    dV = get_gradV(Xdata, Ydata, Vk, lam);
    Vk = Vk - s*dV;
end

Vk - V
verify_V(Vk, V, lam)


%========== auxiliary functions ==========%
function y = get_br_mix(x, V, lam)
% this function returns follower's optimal mixed strategy.
    y = exp(-lam * x' * V)' / sum(exp(-lam * x' * V));
end


function ysamp = get_br_sample(x, V, lam)
% this functions returns a deterministic action sampled from the follower's
% optimal mixed strategy.
    n = size(V, 2);
    y = get_br_mix(x, V, lam);
    ysamp = zeros(n, 1);
    idx = randsample(1:n, 1, true, y);
    ysamp(idx) = 1;
end


function [Xdata, Ydata, Ymix] = sample_strategy(V, lam, N)
% this function samples strategy
    Xdata = [];
    Ydata = [];
    Ymix = [];
    m = size(V, 1);
    for i = 1: N
        x = rand(m, 1);
        x = x / sum(x);
        y = get_br_sample(x, V, lam);
        Xdata = [Xdata x];
        Ydata = [Ydata y];
        Ymix = [Ymix get_br_mix(x, V, lam)];
    end
end


function dV = get_gradV(X, Y, Vk, lam)
% this function computes the gradient of V w.r.t. cross entropy loss l.
    [m, n] = size(Vk);
    N = size(X, 2);
    dV = zeros(m, n);   % each column is dl/dV
    for k = 1: N
        x = X(:, k);
        y = Y(:, k);
        ytmp = get_br_mix(x, Vk, lam);
        dV = dV + (y-ytmp)' .* (lam*x); 
    end
    dV = dV / N;
end


function verify_V(V_pred, V_real, lam)
% this function verifies if V satisfies the randomly generated data.
    N = 10000;
    [xsamp, ~, ysamp] = sample_strategy(V_real, lam, N);
    f_pred = zeros(N, 1);
    f_real = zeros(N, 1);
    policy_err = zeros(N, 1);   % measure distance between 2 policies. 
    
    for i = 1: N
        x = xsamp(:, i);
        y = ysamp(:, i);
        f_real(i) = x' * V_real * y;
        
        y_pred = get_br_mix(x, V_pred, lam);
        f_pred(i) = x' * V_real * y_pred;
        policy_err(i) = sum(y .* log(y./y_pred));
    end
    figure
    plot(f_real)
    hold on
    plot(f_pred)
    plot(f_real-f_pred, 'LineWidth', 3)
    legend('real', 'predict', 'f error');
    
    figure
    plot(policy_err)
    title('test with sampled data');
end
