clc; clear all; close all;
x = load('lgx.dat'); 
y = load('lgy.dat');

[m, n] = size(x);

x = [ones(m, 1), x]; 

figure
pos = find(y == 1); neg = find(y == 0);
plot(x(pos, 2), x(pos,3), '+')
hold on
plot(x(neg, 2), x(neg, 3), 'o')
hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')

theta = zeros(n+1, 1);
g = inline('1.0 ./ (1.0 + exp(-z))'); 
MAX_ITR = 7;

for i = 1:MAX_ITR
    
    h = g(x * theta); 
    grad = x' * (h-y);
    H = x' * diag(h) * diag(1-h) * x;
  
    theta = theta - pinv(H)*grad;
end
theta
plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted')

