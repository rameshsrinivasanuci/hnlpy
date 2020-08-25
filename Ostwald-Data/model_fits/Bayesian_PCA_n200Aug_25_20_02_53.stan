
data {
  int<lower=1> N;                // number of time points
  int<lower=1> P;                // number of components
  matrix[N,P] Y1;                 // Condition 1 EEG matrix of order [N,P]
  matrix[N,P] Y2;                 // Condition 2 EEG matrix of order [N,P]
  matrix[N,P] Y3;                 // Condition 3 EEG matrix of order [N,P]
  matrix[N,P] Y4;                 // Condition 4 EEG matrix of order [N,P]
  int<lower=1> D;              // number of good channels
}
transformed data {
  int<lower=1> M;
  vector[P] mu;
  M  = D*(P-D)+ D*(D-1)/2;  // number of non-zero loadings (Lower triangle of L), lower rectangle area + upper lower triangle
  mu = rep_vector(0.0,P);
}
parameters {    
  vector[M] L_t;   // lower triangular elements of L (latent factors)
  vector<lower=0>[D] L_d;   // diagonal elements of L
  vector<lower=0>[P] psi;         // vector of error variances, truncated at 0 for half-cauchy prior
  real<lower=0>   mu_psi; // hierarchical mean of error variances, truncated at 0 for half-cauchy prior
  real<lower=0>  sigma_psi; // hierarchical std of error variances, truncated at 0 for half-cauchy prior
  real   mu_lt; // hierarchical mean of lower triangular elements of L (latent factors)
  real<lower=0>  sigma_lt; // hierarchical std of lower triangular elements of L (latent factors)
}
transformed parameters{
  cholesky_factor_cov[P,D] L;  //lower triangular factor loadings Matrix 
  cov_matrix[P] Q;   //Covariance matrix of Y
{
  int idx1;
  int idx2;
  real zero;
  idx1 = 0; 
  idx2 = 0;
  zero = 0;
  for(i in 1:P){
    for(j in (i+1):D){
      idx1 = idx1 + 1;
      L[i,j] = zero; //constrain the upper triangular elements to zero 
    }
  }
  for (j in 1:D) {
      L[j,j] = L_d[j]; //Place diagonal elements of latent factor matrix in the matrix
    for (i in (j+1):P) {
      idx2 = idx2 + 1;
      L[i,j] = L_t[idx2]; //Place the lower triangle elements of latent factor matrix in the matrix
    } 
  }
} 
Q=L*L'+diag_matrix(psi); 
}
model {
// the hyperpriors 
   mu_psi ~ cauchy(0, 1);
   sigma_psi ~ cauchy(0,1);
   mu_lt ~ cauchy(0, 1);
   sigma_lt ~ cauchy(0,1);
// the priors 
  L_d ~ cauchy(0,3);
  L_t ~ cauchy(mu_lt,sigma_lt);
  psi ~ cauchy(mu_psi,sigma_psi);
//The likelihood
  for( j in 1:N){
      Y1[j] ~ multi_normal(mu,Q); 
      Y2[j] ~ multi_normal(mu,Q); 
      Y3[j] ~ multi_normal(mu,Q); 
      Y4[j] ~ multi_normal(mu,Q); 
  }
}
