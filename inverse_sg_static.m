% this scirpt implements an iterative algorithm for a static SG to estimate
% the follower's cost matrix V using leader's mixed strategy x and the
% follower's best response yopt.
%       min_x x'*U*yopt  s.t. yopt = argmin_y x'*V*y
% the idea is to solve an optimization problem as inverse_lp.m to estimate
% the cost matrix V.
% the leader's strategy x is chosen randomly.

close all
clearvars
rng(47948);

m = 4;
n = 3;
U = rand(m, n);
V = rand(m, n);
V0 = rand(m,n);     % initial estimate of V
x_hist = [];        % history data
y_hist = [];
err = [];           % error between V_pred and V_real

V_k = V0;
for k = 0: 10
    err = [err; vecnorm(V_k - V, 1)];
    
    % sample x and y
    [xsamp, ysamp] = sample_strategy(V, 10);
    
    % store data
    x_hist = [x_hist xsamp];
    y_hist = [y_hist ysamp];
    
    % perform inverse SG to estimate V
    V_kp1 = inverseSG_l1(V_k, xsamp, ysamp);
    %V_kp1 = inverseSG_l2(V_k, x_hist, y_hist);
    
    V_k = V_kp1;
end
verify_V(V_k, V);
verify_V_with_history(x_hist, y_hist, V_k, V);

figure
plot(err(:, 1))
hold on
plot(err(:, 2))
plot(err(:, 3))

[xsamp, ysamp] = sample_strategy(V, 50000);
V_tmp = inverseSG_l1(V_k, xsamp, ysamp);
V_tmp - V_k
verify_V(V_tmp, V);

% test convex hull things
%testfun(V, V0, x_hist, y_hist)


%====== auxiliary functions ======%
function y = get_br(x, V)
    n = size(V, 2);
    y = zeros(n, 1);
    [~, idx] = min(x'*V);
    y(idx) = 1;
end


function [xsamp, ysamp] = sample_strategy(V, N)
    [m, n] = size(V);
    xsamp = zeros(m, N);
    ysamp = zeros(n, N);
    for i = 1: N
        x = rand(m, 1);
        x = x / sum(x);
        y = get_br(x, V);
        xsamp(:, i) = x;
        ysamp(:, i) = y;
    end
end


function [Vopt] = inverseSG_l1(V0, xsamp, ysamp)
% formulate l1 norm as objective. Z = [V1,...,Vn,t1,...,tn]
    [m, n] = size(V0);
    N = size(xsamp, 2);
    A = [];
    b = [];
    % l1 norm reformulation
    A = [A; kron(eye(n), eye(m)) kron(eye(n), -eye(m)); ...
            kron(eye(n), -eye(m)) kron(eye(n), -eye(m))];
    b = [b; reshape(V0, [m*n,1]); -reshape(V0, [m*n,1])];
    % KKT condition
    for i = 1: N
        x = xsamp(:, i);
        y = ysamp(:, i);
        a1 = ones(n,1) * reshape(y'.*x, [1,m*n]) - kron(eye(n), x');
        A = [A; a1 zeros(n, m*n)];
        b = [b; zeros(n, 1)];
    end
    
    f = [zeros(1,m*n) ones(1,m*n)];
    Zopt = linprog(f,A,b);
    Vopt = reshape(Zopt(1:m*n), [m,n]);
end


function [Vopt] = inverseSG_l2(V0, xsamp, ysamp)
% formulate l2 norm as objective. Z = [V1, ..., Vn]
    [m, n] = size(V0);
    N = size(xsamp, 2);
    A = [];
    b = [];
    % KKT condition
    for i = 1: N
        x = xsamp(:, i);
        y = ysamp(:, i);
        a1 = ones(n,1) * reshape(y'.*x, [1,m*n]) - kron(eye(n), x');
        A = [A; a1];
        b = [b; zeros(n, 1)];
    end
    H = eye(m*n);
    f = -reshape(V0, [m*n,1]);
    Zopt = quadprog(H,f,A,b);
    Vopt = reshape(Zopt(1:m*n), [m,n]);
end


function verify_V(V_pred, V_real)
% this function verifies if V satisfies the randomly generated data.
    N = 1000000;
    [xsamp, ysamp] = sample_strategy(V_real, N);
    f_pred = zeros(N, 1);
    f_real = zeros(N, 1);
    n_pred = zeros(N, 1);   % 1 means correct, 0 means wrong
    
    for i = 1: N
        x = xsamp(:, i);
        y = ysamp(:, i);
        f_real(i) = x' * V_real * y;
        y_pred = get_br(x, V_pred);
        if all(y - y_pred == 0)
            n_pred(i) = 1;
        end
        f_pred(i) = x' * V_real * y_pred;
    end
    figure
    plot(f_real)
    hold on
    plot(f_pred)
    plot(f_real-f_pred, 'LineWidth', 3)
    title('test with sampled data');
    legend('real', 'predict', 'error');
    fprintf('error rate: %.5f.\n', 1-sum(n_pred)/N);
end


function verify_V_with_history(x_hist,y_hist, V_pred, V_real)
% this function verifies if V satsifies the history data.
    N = size(x_hist, 2);
    f_pred = zeros(N, 1);
    f_real = zeros(N, 1);
    n_pred = zeros(N, 1);   % 1 means correct, 0 means wrong
    
    for i = 1: N
        x = x_hist(:, i);
        y = y_hist(:, i);
        f_real(i) = x' * V_real * y;
        y_pred = get_br(x, V_pred);
        if all(y - y_pred == 0)
            n_pred(i) = 1;
        end
        f_pred(i) = x' * V_real * y_pred;
    end
    figure
    plot(f_real)
    hold on
    plot(f_pred)
    plot(f_real-f_pred, 'LineWidth', 3)
    title('test with history data');
    legend('real', 'predict', 'error');
    fprintf('error rate: %.5f.\n', 1-sum(n_pred)/N);
end


function testfun(V_real, V0, xsamp, ysamp)
    if isempty(xsamp)
        N = 1000;
        [xsamp, ysamp] = sample_strategy(V_real, N);
    else
        N = size(xsamp, 2);
    end
    [m, n] = size(V_real);
    
    % process each partition and obtain convex hull
    X_hull = cell(n, 1);
    Y_hull = cell(n, 1);
    for i = 1: N
        x = xsamp(:, i);
        y = ysamp(:, i);
        idx = find(y);
        if inhull(X_hull{idx}, x) == 0
            X_hull{idx} = [X_hull{idx} x];
            Y_hull{idx} = [Y_hull{idx} y];
            % eliminate redundant vector only for N > n
            if size(X_hull{idx}, 2) > m
                k = convhulln(X_hull{idx}', {'QJ', 'QbB'});   % transpose points
                X_hull{idx} = X_hull{idx}(:, unique(k));
                Y_hull{idx} = Y_hull{idx}(:, unique(k));
            end
        end
    end
    
    % concatenate data
    x_data = [];
    y_data = [];
    for i = 1: n
        x_data = [x_data X_hull{idx}];
        y_data = [y_data Y_hull{idx}];
    end
    V_test = inverseSG_l1(V0, x_data, y_data);
    verify_V(V_real, V0);
    
    function [flag] = inhull(hull, xx)
        if isempty(hull)
            flag = 0;
            return
        end
    
        % formulate LP to determine if x is in the hull
        NN = size(hull, 2);
        Aeq = [hull; ones(1, NN)];
        beq = [xx; 1];
        lb = zeros(NN, 1);
        f = zeros(NN, 1);
        options = optimoptions('linprog','Display','final');
        [~, ~, exitflag, ~] = linprog(f,[],[],Aeq,beq,lb,[],options);
        if exitflag ~= 1
            flag = 0;
        else
            flag = 1;
        end
    end
end
