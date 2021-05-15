function [ reg ] = reg_term(X,Sbasis,n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
X=reshape(X,n^2,size(Sbasis,1));
reg=X*Sbasis;
reg=reg(:);
end

