% this script implements an algorithm for inverse dynamic regularized SG.
close all
clearvars
rng(4738);

dims = 5;
dima = 4;
dimb = 3;
ga = rand(dims, dima, dimb);
gb_real = rand(dims, dima, dimb);
gaf = zeros(dims, 1);               % terminal cost
gbf = zeros(dims, 1);
gb0 = rand(dims, dima, dimb);       % initial estimate of V
p = rand(dims, dims, dima, dimb);   % p(s'|s,a,b)
p = normalize_p(p);
gam = 0.5;
lam = 10;
N = 50;         % sample size
GD_ITER = 200;
GD_STEP = 1e-2;
EPOCH = 10;
T = 5;
dt = 3;
gb_epoch = cell(EPOCH + 1);
gb_epoch{1} = gb0;

Dall = ground_truth(ga, gb_real, gaf, gbf, p, lam, gam, T, dt);

for iter = 1: EPOCH
    va = cell(T, 1);
    vb = cell(T, 1);
    va{T} = gaf;
    vb{T} = gbf;
    gb = gb_epoch{iter};
   
    % perform DP to estimate gb
    for t = T-1: -1: 1
        if mod(t, dt) ~= 0  % follower always play 1
            [va{t}, vb{t}] = update_va_vb_1(ga, gb, va{t+1}, vb{t+1}, p, gam);
            %vb{t} = update_vb_1(gb, vb{t+1}, p, gam, xopt_idx);
        else
            Dt = sample_strategy(t, N, Dall);
            gb = estimate_gb(gb, vb{t+1}, p, gam, lam, Dt, GD_ITER, GD_STEP);
            [va{t}, vb{t}] = update_va_vb_2(ga, gb, va{t+1}, vb{t+1}, p, gam, lam);
            %vb{t} = update_vb_2(gb, vb{t+1}, p, gam, lam, xopt);
        end
    end
    gb_epoch{iter+1} = gb;
    fprintf('epoch %d finished.\n', iter);
end

validate_gb(gb, gb_real);

%========== auxiliary functions ==========%
function [p] = normalize_p(p)
    % normalize the first dimension.
    [dimS, ~, dimA, dimB] = size(p);
    tmp = squeeze(sum(p, [1]));
    for i = 1: dimS    % for s
        for j = 1: dimA    % for a
            for k = 1: dimB    % for b
                for l = 1: dimS    % for s'
                    p(l, i,j,k) = p(l, i,j,k) / tmp(i,j,k);
                end
            end
        end
    end
end


function [data] = ground_truth(ga, gb, gaf, gbf, p, lam, gam, T, dt)
% this function generates ground truth data.
% data(t) = {s1{x,y*}, s2{x,y*}, ..., sn{x,y*}}
    [dims, dima, dimb] = size(ga);
    data = cell(T-1, 1);
    va = cell(T, 1);
    vb = cell(T, 1);
    va{T} = gaf;
    vb{T} = gbf;
    N = 2000;        % sample N=100 data at each time instance
    for t = T-1: -1: 1
        data_t = cell(dims, 1);
        if mod(t, dt) ~= 0  % only update value function
            [va{t}, vb{t}] = update_va_vb_1(ga, gb, va{t+1}, vb{t+1}, p, gam);
        else        % sample for each state s
            for s = 1: dims
                xsamp = rand(dima, N);
                xsamp = xsamp ./ sum(xsamp, 1);
                % formulate cB to get sample data
                cB = squeeze(gb(s,:,:));
                for i = 1:dima
                    for j = 1: dimb
                        cB(i,j) = cB(i,j) + gam * sum(p(:,s,i,j)'*vb{t+1});
                    end
                end
                % sample yopt for each x
                ysamp = zeros(dimb, N);
                for i = 1: N
                    ymix = exp(-lam*xsamp(:,i)'*cB)' / sum(exp(-lam*xsamp(:,i)'*cB));
                    ytmp = zeros(dimb, 1);
                    idx = randsample(1:dimb, 1, true, ymix);
                    ytmp(idx) = 1;
                    ysamp(:, i) = ytmp;
                end
                data_t{s} = {xsamp, ysamp};
            end
            [va{t}, vb{t}] = update_va_vb_2(ga, gb, va{t+1}, vb{t+1}, p, gam, lam);
        end
        data{t} = data_t;
    end
end


function [D] = sample_strategy(t, N, data)
% this function randomly samples strategy from ground truth data.
    dims = length(data{t});
    D = cell(dims, 1);
    for i = 1: dims
        Ns = floor(N / dims);
        xdata = data{t}{i}{1};
        ydata = data{t}{i}{2};
        l = size(xdata, 2);
        idx = randsample(l, Ns);
        D{i} = {xdata(:, idx); ydata(:, idx)};
    end
end


function [va_t, vb_t] = update_va_vb_1(ga, gb, va_tp1, vb_tp1, p, gam)
% this function updates va when the follower does not make decision.
    [dims, dima, ~] = size(ga);
    va_t = zeros(dims, 1);
    vb_t = zeros(dims, 1);
    % formulate cA and cB, then update value
    for s = 1: dims
        cA = ga(s,:,1);
        for i = 1:dima
            cA(i) = cA(i) + gam * sum(p(:,s,i,1)'*va_tp1);
        end
        % update value
        [va_t(s), x_idx] = min(cA);
        vb_t(s) = gb(s,x_idx,1) + gam * sum(p(:,s,x_idx,1)'*vb_tp1);
    end
end


% function vb_t = update_vb_1(gb, vb_tp1, p, gam, x_idx)
% % this function updates vb when the follower does not make decision.
%     [dims, ~, ~] = size(gb);
%     vb_t = zeros(dims, 1);
%     for s = 1: dims
%         vb_t(s) = gb(s,x_idx(s),1) + gam * sum(p(:,s,x_idx(s),1)'*vb_tp1);
%     end
% end


function [gb_new] = estimate_gb(gb, vb_tp1, p, gam, lam, Dt, ITER, step)
% this function estimates gb using GD. subtraction is needed to get real gb
    [dims, dima, dimb] = size(gb);
    gb_new = zeros(size(gb));
    for i = 1: dims
        gb_tmp = do_GD(squeeze(gb(i,:,:)), Dt{i}{1}, Dt{i}{2});
        gb_tmp1 = zeros(dima, dimb);
        for a = 1: dima
            for b = 1: dimb
                gb_tmp1(a,b) = gam * sum(p(:, i,a,b) .* vb_tp1);
            end
        end
        gb_new(i,:,:) = gb_tmp - gb_tmp1;   % subtract value
    end
    
    function [V] = do_GD(V, X, Y)
    % inner functio to perform GD
        N = size(X, 2);
        for k = 1: ITER
            % get gradient
            dV = zeros(dima, dimb);
            for j = 1: N
                x = X(:, j);
                y = Y(:, j);
                ytmp = exp(-lam * x' * V)' / sum(exp(-lam * x' * V));
                dV = dV + (y-ytmp)' .* (lam*x); 
            end
            dV = dV / N;
            V = V - step*dV;
        end
    end
end


function [va_t, vb_t] = update_va_vb_2(ga, gb, va_tp1, vb_tp1, p, gam, lam)
% this function updates va when follower makes the decision.
% use GD to find xopt by solving an constraint optimization.
    [dims, dima, dimb] = size(ga);
    va_t = zeros(dims, 1);
    vb_t = zeros(dims, 1);
    % formulate cA and cB
    for s = 1: dims
        cA = squeeze(ga(s,:,:));
        cB = squeeze(gb(s,:,:));
        for i = 1:dima
            for j = 1: dimb
                cA(i,j) = cA(i,j) + gam * sum(p(:,s,i,j)'*va_tp1);
                cB(i,j) = cB(i,j) + gam * sum(p(:,s,i,j)'*vb_tp1);
            end
        end
    
        % formulate leader's problem to solve xopt
        fun = @(x) myobj(x, cA, cB, lam);
        x0 = rand(dima, 1);
        x0 = x0 / sum(x0);
        options = optimoptions('fmincon','Display','off');
        xopt = fmincon(fun,x0, [],[], ones(1,dima),1, zeros(dima,1),[], [], options);
        
        % update va_t and vb_t
        yopt = exp(-lam * xopt' * cB)' / sum(exp(-lam * xopt' * cB));
        va_t(s) = xopt' * cA * yopt;
        vb_t(s) = xopt' * cB * yopt;
    end
    
    function [f] = myobj(x, cA, cB, lam)
        %[dima, dimb] = size(cA);
        f = 0;
        tmp = 0;
        for k = 1: dimb
            f = f + x' * cA(:,k) * exp(-lam*x'*cB(:,k));
            tmp = tmp + exp(-lam*x'*cB(:,k));
        end
        f = f / tmp;
    end
end


% function [vb_t] = update_vb_2(gb, vb_tp1, p, gam, lam, xopt)
% % this function updates vb when follower makes the decision.
% % xopt = dima * dims
%     % formulate cB and find yopt, then compute vb.
%     [dims, dima, dimb] = size(gb);
%     vb_t = zeros(dims, 1);
%     for s = 1: dims
%         cB = squeeze(gb(s,:,:));
%         for i = 1:dima
%             for j = 1: dimb
%                 cB(i,j) = cB(i,j) + gam * sum(p(:,s,i,j)'*vb_tp1);
%             end
%         end
%         yopt = exp(-lam * xopt' * cB)' / sum(exp(-lam * xopt' * cB));
%         vb_t(s) = xopt' * cB * yopt;
%     end
% end


function validate_gb(gb_pred, gb_real)
% this function validate estimated gb with real gb.
    [dims, dima, dimb] = size(gb_real);
    for i = 1: dims
        squeeze(gb_pred(i,:, :) - gb_real(i,:,:))
    end
end