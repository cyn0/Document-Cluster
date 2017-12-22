load('C:\Files of Krishnan\Files of Krishnan\Pattern Recognition\Project\20Newsgroups.mat')

X = [fea(1:100, :); fea(800:900, :); fea(1674:1774, :); fea(2659:2759, :); fea(3641:3741, :); fea(4604:4704, :); fea(5592:5692, :); fea(6567:6667, :); fea(7557:7657, :); fea(8553:8653, :)]';
y = [gnd(1:100, :); gnd(800:900, :); gnd(1674:1774, :); gnd(2659:2759, :); gnd(3641:3741, :); gnd(4604:4704, :); gnd(5592:5692, :); gnd(6567:6667, :); gnd(7557:7657, :); gnd(8553:8653, :)]';

%lambda = 1;
meu = 0.001;

[rows, cols] = size(X);

epsilons = [3 2 1.5 1 0.8]';
approx = [2 3 5 7 10]';
error = zeros(5, 1);

for app = 1:5
    
r = approx(app, 1);

approx_error_nnsc = zeros(5, 1);
error_nnsc = zeros(5, 1);

lambda_v = zeros(5, 1);  

epsilon_1 = epsilons(app, 1);
epsilon_2 = epsilons(app, 1);

k = 1;

for lambda = 0.001 : 0.4: 1.603
  
A = randn(rows, r);
S = 0.01 * ones(r, cols);

epsilon1 = 0;
epsilon2 = 0;


A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

while true
    A_new = (A - meu * (A * S - X) * S');
    A_new = (A_new + abs(A_new))/2;
    
    for i = 1:r
        A_new(:, i) = A_new(:, i)/norm(A_new(:, i)); %we normalise the ith column of A_new and put the new column into A at the same position as i
    end
    
    S_new = S .* (A_new' * X) ./ (A_new' * A_new * S + lambda);
    S_new = (S_new + abs(S_new))/2;
    
    epsilon1 = norm(A_new - A, 'fro') / sqrt(rows * r);
    epsilon2 = norm(S_new - S, 'fro') / sqrt(r * cols);
    
    if (epsilon1 < epsilon_1 && epsilon2 < epsilon_2) %when the new A is not very different from old A
        break; %no point in continuing
    end
    
    A = A_new;
    S = S_new;  
end

sum = 0;

for i = 1:cols
    sum = sum + norm(S_new(:, i), 1);
end 

approx_error_nnsc(k, 1) = norm(X - A_new * S_new, 'fro');

error_nnsc(k, 1) = sum;
lambda_v(k, 1) = lambda;

k = k + 1;
end

figure;

subplot(1,3,1);
plot(lambda_v, approx_error_nnsc);
title('Error vs Lambda');
xlabel('Lambda');
ylabel('Error');

subplot(1,3,2);
plot(lambda_v, error_nnsc);
title('Sparsity vs Lambda');
xlabel('Lambda');
ylabel('Sparsity');

subplot(1,3,3);
plot(error_nnsc, approx_error_nnsc);
title('Error vs Sparsity');
xlabel('Sparsity');
ylabel('Error');


[min1, I1] = min(approx_error_nnsc);
approx_error_nnsc(I1, 1) = Inf;

[min2, I2] = min(approx_error_nnsc);
lambda_final = lambda_v(I2, 1);

A = randn(rows, r);
S = 0.01 * ones(r, cols);

epsilon1 = 0;
epsilon2 = 0;

A_new = zeros(rows, r);
S_new = zeros(r, cols);

A = (A + abs(A))/2;
S = (S + abs(S))/2;

for i = 1:r
   A(:, i) = A(:, i)/norm(A(:, i)); %we normalise the ith column of A
end

while true
    A_new = (A - meu * (A * S - X) * S');
    A_new = (A_new + abs(A_new))/2;
    
    for i = 1:r
        A_new(:, i) = A_new(:, i)/norm(A_new(:, i)); %we normalise the ith column of A_new and put the new column into A at the same position as i
    end
    
    S_new = S .* (A_new' * X) ./ (A_new' * A_new * S + lambda);
    S_new = (S_new + abs(S_new))/2;
    
    epsilon1 = norm(A_new - A, 'fro') / sqrt(rows * r);
    epsilon2 = norm(S_new - S, 'fro') / sqrt(r * cols);
    
    if (epsilon1 < epsilon_1 && epsilon2 < epsilon_2) %when the new A is not very different from old A
        break; %no point in continuing
    end
    
    A = A_new;
    S = S_new;  
end

cluster = zeros(1, cols);

for i = 1:cols
    [M, I_max] = max(S_new(:, i));
    
    cluster(1, i) = I_max;
end

for i = 1:10
    [mode_F, F] = mode(cluster(1, ((i - 1) * 100 + 1): i * 100));
    
    error(app, 1) = error(app, 1) + (100 - F);
end

end

figure;
plot(approx, error);
title('Misclustering vs Number of clusters');
xlabel('Number of clusters');
ylabel('Misclustering');
