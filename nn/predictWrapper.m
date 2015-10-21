clear;close all;clc
arglist = argv()

fprintf('Loading Data ...\n')
load(theta_values.m)
load(arglist{1})

pred = predict(Theta1,Theta2,X)
fprintf('Finished Predicting ...\n')
