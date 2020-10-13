function [Y] = CsrSigmoid(X)
    Y = 1./(1 + exp(-X));
end