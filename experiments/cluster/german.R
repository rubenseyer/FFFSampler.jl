library(rstan)
data <- read.table("german.data-numeric")
X <- scale(data[,-ncol(data)])
y <- data[,ncol(data)] - 1

model_code <- "
  data {
    int<lower=0> N;
    int<lower=0> D;
    matrix[N,D] x;
    array[N] int<lower=0, upper=1> y;
  }
  parameters {
    real alpha;
    vector[D] beta;
  }
  model {
    alpha ~ normal(0,100);
    beta ~ normal(0,100);
    y ~ bernoulli_logit(alpha + x * beta);
  }
"

fit <- stan(model_name="lr", model_code=model_code,
            data=list(N=nrow(X),D=ncol(X),x=X,y=y), iter=20000, chains=10, thin=10, seed=1234)
check_hmc_diagnostics(fit)

draws <- as.matrix(fit)
lags <- apply(draws, 2, \(v) abs(acf(v,1,plot=F)[1][[1]]))
write.table(draws[,-ncol(draws)], "german.reference", row.names=F, col.names=T)
              