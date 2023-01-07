% this script tests the inverse problem of the following LP:
% min c'*x  s.t. sum(x) = 1, x >= 0.
% the inverse problem estimates c given the optimal solution xopt.
% note that there is no parameter in the constraints.

close all 
clearvars
%rng(200);

n = 200;
c = randn(n, 1);
c0 = rand(n, 1);    % initial guess
Aeq = ones(1, n);
beq = 1;
lb = zeros(n, 1);
ub = ones(n, 1);
xopt = linprog(c,[],[],Aeq,beq,lb,ub);  % get xopt

% find new c using different objective function
copt_l2 = l2norm(xopt, c0);
copt_l1 = l1norm(xopt, c0);
copt_linf = linfnorm(xopt, c0);
%copt = l1norm_v2(xopt, c0);
copt_l1cc = l1norm_v3(xopt, c0);

% check if xopt is the optimal solution of new c
xnew = linprog(copt,[],[],Aeq,beq,lb,ub);
find(xopt-xnew ~= 0)
%copt-c0


%======= auxiliary functions =======%
function [copt] = l2norm(xopt, c0)
% use l2 norm as objective function
    n = size(c0, 1);
    A = ones(n,1) * xopt' - eye(n);
    b = zeros(n, 1);
    H = eye(n);
    f = -c0;
    %lb = zeros(n,1); ub = ones(n,1);
    copt = quadprog(H,f,A,b);
end


function [copt] = l1norm(xopt, c0)
% use l1 norm as objective function, Z = [c, t]
    n = size(c0, 1);
    A = ones(n,1) * xopt' - eye(n);
    A = [A zeros(n)];
    b = zeros(n, 1);
    A = [A; eye(n) -eye(n); -eye(n) -eye(n)];
    b = [b; c0; -c0];
    f = [zeros(1,n) ones(1,n)]; 
    Zopt = linprog(f,A,b);
    copt = Zopt(1:n);
end


function [copt] = linfnorm(xopt, c0)
% use linf norm as objective function. Z = [c, t]
    n = size(c0, 1);
    A = ones(n,1) * xopt' - eye(n);
    A = [A zeros(n,1)];
    b = zeros(n, 1);
    A = [A; eye(n) -ones(n,1); -eye(n) -ones(n,1)];
    b = [b; c0; -c0];
    f = [zeros(1,n) 1]; 
    Zopt = linprog(f,A,b);
    copt = Zopt(1:n);
end


function [copt] = l1norm_v2(xopt, c0)
% use l1 norm and another formulation to test. Z = [c,mu,t]
    n = size(c0, 1);
    A = [-eye(n) -ones(n,1) zeros(n); eye(n) zeros(n,1) -eye(n); -eye(n) zeros(n,1) -eye(n)];
    b = [zeros(n,1); c0; -c0];
    Aeq = [xopt' xopt'*ones(n,1) zeros(1,n)];
    beq = 0;
    f = [zeros(1,n+1) ones(1,n)]; 
    Zopt = linprog(f,A,b,Aeq,beq);
    copt = Zopt(1:n);
end


function [copt] = l1norm_v3(xopt, c0)
% use l1 norm and strict complementary slackness formulation. Z = [c,t]
    eps = 1e-5;
    n = size(c0, 1);
    A = [eye(n) -eye(n); -eye(n) -eye(n)];
    b = [c0; -c0];
    % strict cc formulation
    Aeq = [];
    beq = [];
    for i = 1: n
        a = zeros(1, n);
        a(i) = 1;
        if xopt(i) == 0
            A = [A; xopt'-a zeros(1,n)];
            b = [b; -eps];
        else
            Aeq = [Aeq; xopt'-a zeros(1,n)];
            beq = [beq; 0];
        end
    end
    f = [zeros(1,n) ones(1,n)]; 
    Zopt = linprog(f,A,b,Aeq,beq);
    copt = Zopt(1:n);
end
