close all;
clear all; 
clc;
warning('off','all');

x = load('x.dat');
y = load('y.dat');

m = length(x);

figure;
 scatter (x, y, 7, 'b', 'filled');
hold on;

x = [ones(m,1) x];
max_iter = 1500;
alpha = 0.007;
theta = [0  0];

for iter=1:max_iter    
  for i=1:m
    grad = (x(i,:)*theta' - y(i,:)) .* x(i,:);
    theta  = theta - alpha * grad ;   
  end;   
end;

theta

plot(x(:,2), x*theta' ,'r', 'linewidth', 3)

[1,3.5]*theta'
[1,  7]*theta'

