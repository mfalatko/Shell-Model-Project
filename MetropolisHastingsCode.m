
%logit_model.m

 clear all,clc

 n=10000;
 k=4;   

 randn('seed', 12345);  rand('seed', 67890);

 truebeta = [1 -5.5 1 3]'; 
 X = [ ones(n,1) randn(n,k-1)*0.1 ];  
 Y = binornd(1,1./(1+exp(-X*truebeta)));
 bo=zeros(k,1); 

  err=inf; b=bo;                        %initial guess 
 while norm(err)>10^(-3)
     p=1./(1+exp(-X*b));
     g=X'*(Y-p)-(b-bo)/100;
     H=-X'*diag(p.^2.*(1./p-1))*X-eye(k)/100;
     err=H\g;                 % make Newton-Raphson correction
     b=b-err;                             % update Newton guess
 end


 proposal_sd=sqrt(0.01);

logf=@(b)(-.5*(b-bo)'*(b-bo)/100-Y'*log(1+exp(-X*b)) - (1-Y)'*(X*b+log(1+exp(-X*b))));
 alpha=@(x,y)min(1,exp(logf(y)-logf(x)));

 df=10; 
T=10^4; 
data=nan(T,k); 
 for t=1:T

      b_star = b + sqrt(0.01)*randn(k,1);
     if rand<alpha(b,b_star)
         b=b_star;
     end
     data(t,:)=b';
 end
 b_hat=mean(data)
Cov_hat=cov(data)