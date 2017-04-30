
##*Here we consider the lung cancer data in file* lung-cancer-1.csv *and perform a series of learning tasks.* 
  library(class)
  library(ElemStatLearn)
  library(kernlab)
  library(MASS)
  library(e1071)

  lung.data <- read.csv('lung-cancer-1.csv')
  str(lung.data,list.len=5)

#  We have found the class attributes, they are in the first column, and are labeled **Y**.  
  

  factor(lung.data$Y)

#  Now we know that there are *four* types of cancers in this dataset, corresponding to classes *k=0,1,2,3*.  
  
#  To determine the prior probabilities, we need to count up the occurances of each class in the 197 observations:
  

  c.freq <- as.data.frame(table(lung.data$Y))
  c.freq$per <- round(c.freq$Freq/sum(c.freq$Freq)*100,2)
  names(c.freq)[1] = 'class'
  names(c.freq)[3] = '%'
  c.freq


#  The priors are the frequencies of each class.
  
#  $pi$_0 = ...

#  Obtain the Quadratic Discriminant Analysis classifier for this dataset, comment extensively:

#  Let's standardize the data first:
 
     standard <-function(xx)
   {
     n <- nrow(xx)
     p <- ncol(xx)
     aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
     bb  <- sqrt(matrix(rep(apply(xx,2,var), n), ncol=p, byrow=TRUE))
     return((xx-aa)/bb)
     }

     xx <- standard(lung.data[,-1])
     lung.data <- cbind(lung.data[,1],xx)
     colnames(lung.data)[1] <- 'Y'



# mu0 <- numeric(p)
# mu1 <- numeric(p)
# mu2 <- numeric(p)
# mu3 <- numeric(p)

#make a matrix of the four mu values
  mu <- matrix(0,nrow=4,ncol=p)

#figure out which rows correspond to each class
  id.zero    <- which(lung.data$Y == 0)
  n.zero     <- length(id.zero)
  
  id.one    <- which(lung.data$Y == 1)
  n.one     <- length(id.one)
    
  id.two    <- which(lung.data$Y == 2)
  n.two     <- length(id.two)
    
  id.three    <- which(lung.data$Y == 3)
  n.three     <- length(id.three)
    
#split up the data into four n x p matrices, where each matrix is just the rows from one class  
  
  lung.data.0 <- matrix(0,nrow=n,ncol=p)
  lung.data.1 <- matrix(0,nrow=n,ncol=p)
  lung.data.2 <- matrix(0,nrow=n,ncol=p)
  lung.data.3 <- matrix(0,nrow=n,ncol=p)
  
#here we can loop through to create both our mu vectors and fill in our four separate data matrices
  for(i in 1:n.zero)
  {
    lung.data.0[id.zero[i],] <- lung.data.0[id.zero[i],] + as.matrix(lung.data[id.zero[i],-1])
    mu[1,] <- mu[1,] + as.matrix(lung.data[id.zero[i],-1])
  }
  
  mu[1,] <- mu[1,]/n.zero
     
  for(i in 1:n.one)
  {
    lung.data.1[id.one[i],] <- lung.data.1[id.one[i],] + as.matrix(lung.data[id.one[i],-1])
    mu[2,] <- mu[2,] + as.matrix(lung.data[id.one[i],-1])
  }
  
  mu[2,] <- mu[2,]/n.one
     
  for(i in 1:n.two)
  {
    lung.data.2[id.two[i],] <- lung.data.2[id.two[i],] + as.matrix(lung.data[id.two[i],-1])
    mu[3,] <- mu[3,] + as.matrix(lung.data[id.two[i],-1])
  }
  
  mu[3,] <- mu[3,]/n.two
     
  for(i in 1:n.three)
  {
    lung.data.3[id.three[i],] <- lung.data.3[id.three[i],] + as.matrix(lung.data[id.three[i],-1])
    mu[4,] <- mu[4,] + as.matrix(lung.data[id.three[i],-1])
  }
  
  mu[4,] <- mu[4,]/n.three
  
#initialize the sigma's    
  sig0 <- matrix(0,nrow=p,ncol=p)
  sig1 <- matrix(0,nrow=p,ncol=p)
  sig2 <- matrix(0,nrow=p,ncol=p)
  sig3 <- matrix(0,nrow=p,ncol=p)
  
#calculate the sigma for each class  
  sig0 <- cov(lung.data.0[,])
  sig1 <- cov(lung.data.1[,])
  sig2 <- cov(lung.data.2[,])
  sig3 <- cov(lung.data.3[,])
  
#calculate the inverse for each sigma
  invSig0 <- solve(sig0)
  invSig1 <- solve(sig1)
  invSig2 <- solve(sig2)
  invSig3 <- solve(sig3)

#  Unfortunately, the sigma's are singular and cannot be inverted.  This means we are unable to use QDA as a learning algorithm.
  
# Similarly, we run into errors when trying to use the qda code in R:

  library(class)
  library(ElemStatLearn)
  library(kernlab)
  library(MASS)
  library(e1071)

  n <- nrow(lung.data)
  ntr.qda <- round(0.75*n)
  ite.qda <- ntr.qda+1

  xy.tr.qda <- lung.data[1:ntr.qda]
  xy.te.qda <- setdiff(lung.data,xy.tr.qda)
  
  qda.lung <- qda(Y~., data=xy.tr.qda)
  yhat.qda <- predict(qda.lung, lung.data[ite.qda:n,-1])$class
  qda.table <- table(lung.data[ite.qda:n,]$Y, yhat.qda)
  err.qda <- 1-sum(diag(table(lung.data$Y, yhat.qda)))/n

  
#Comparison of Predictive performances.  Consider four classifiers: **1NN**, **3NN**, **LDA**, and **Naive Bayes**.  For each of these classifiers use a *stratified hold out* split with 75% Training and 25% test.  (Stratified hold out means that the training and test data must be consistent with the prior's found above).  Perform R=49 splits, computing the test error for each split.


  R <- 49 #number of splits
  ntest = 0.75 #training split
    
  pos <- 1  #position of Y column
  
  err.str <- matrix(0, nrow=R, ncol = 8)  #error grid with avg tr err, avg te err for each 4 methods
  
  for (r in 1:R)
  {
    id.zero    <- which(lung.data$Y == 0)
    n.zero     <- length(id.zero)
    id.zero.tr <- sample(sample(sample(id.zero)))[1:round(ntest*n.zero)]
    id.zero.te <- setdiff(id.zero, id.zero.tr)
  
    id.one    <- which(lung.data$Y == 1)
    n.one     <- length(id.one)
    id.one.tr <- sample(sample(sample(id.one)))[1:round(ntest*n.one)]
    id.one.te <- setdiff(id.one, id.one.tr)
  
    id.two    <- which(lung.data$Y == 2)
    n.two     <- length(id.two)
    id.two.tr <- sample(sample(sample(id.two)))[1:round(ntest*n.two)]
    id.two.te <- setdiff(id.two, id.two.tr)
    
    id.three    <- which(lung.data$Y == 3)
    n.three     <- length(id.three)
    id.three.tr <- sample(sample(sample(id.three)))[1:round(ntest*n.three)]
    id.three.te <- setdiff(id.three, id.three.tr)
  
    xy.tr <- lung.data[c(id.zero.tr,id.one.tr,id.two.tr,id.three.tr), ]
    xy.te <- lung.data[c(id.zero.te,id.one.te,id.two.te,id.three.te), ]
    ntr <- nrow(xy.tr)
    nte <- nrow(lung.data) - ntr

    yhat.1NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=1)
    err.1NN.tr <- 1-sum(diag(table(xy.tr$Y, yhat.1NN.tr)))/ntr
    yhat.1NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=1)
    err.1NN.te <- 1-sum(diag(table(xy.te$Y, yhat.1NN.te)))/nte
      
    err.str[r,1] <- err.1NN.tr
    err.str[r,2] <- err.1NN.te

    yhat.3NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=3)
    err.3NN.tr <- 1-sum(diag(table(xy.tr$Y, yhat.3NN.tr)))/ntr
    yhat.3NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=3)
    err.3NN.te <- 1-sum(diag(table(xy.te$Y, yhat.3NN.te)))/nte
      
    err.str[r,3] <- err.3NN.tr
    err.str[r,4] <- err.3NN.te
    
    lda.xy <- lda(Y~., data=xy.tr)  
    yhat.lda.tr <- predict(lda.xy, xy.tr[,-pos])$class
    yhat.lda.te <- predict(lda.xy, xy.te[,-pos])$class
    err.lda.tr <- 1-sum(diag(table(xy.tr$Y, yhat.lda.tr)))/ntr
    err.lda.te <- 1-sum(diag(table(xy.te$Y, yhat.lda.te)))/nte
      
    err.str[r,5] <- err.lda.tr
    err.str[r,6] <- err.lda.te
    
    nb.xy <- naiveBayes(Y~., data=xy.tr)
    yhat.nb.tr <- predict(nb.xy, xy.tr[,-1], type='raw')
    yhat.nb.tr <- colnames(yhat.nb.tr)[apply(yhat.nb.tr,1,which.max)]
    yhat.nb.te <- predict(nb.xy, xy.te[,-1], type='raw')
    yhat.nb.te <- colnames(yhat.nb.te)[apply(yhat.nb.te,1,which.max)]
    err.nb.tr <- 1-sum(diag(table(xy.tr$Y, yhat.nb.tr)))/ntr
    err.nb.te <- 1-sum(diag(table(xy.te$Y, yhat.nb.te)))/nte
    
    err.str[r,7] <- err.nb.tr
    err.str[r,8] <- err.nb.te
    
#    if (r%%1==0)  cat('\n', round(100*r/R,0),'completed\n')
}

#  Now we have a grid of average training and test errors for each of our splits.  Let's make a boxplot of the errors:
  

  boxplot(err.str[,c(2,4,6,8)], col=c(2,3,4,5), names=c('1nn','3nn','LDA','NB'),main='stratified split',ylab='test errors')



#  Here they are in table form:
  

  avg.err <- c(0,0,0,0,0,0,0,0)
  for(i in 1:8)
  {
    avg.err[i] <- mean(err.str[,i])
  }
  avg.err.table <- matrix(0,nrow=4,ncol=2)
  count=1
  for(i in 1:4)
  {
    for(j in 1:2)
    {
      avg.err.table[i,j] <- avg.err[count]
      count <- count+1
    }
  }
  rownames(avg.err.table) <- c('1nn','3nn','lda','NB')
  colnames(avg.err.table) <- c('avg. tr. err','avg. te. err')
  avg.err.table


#  And a colorful point plot of the average test errors for good measure:
  

  plot(1:8, avg.err, ylab='Average prediction error', xlab='Method (Classifier)',xlim=c(0,9), ylim=c(0.90*min(avg.err),1.10*max(avg.err)))
  text(1:8, avg.err, col=c(2,3,4,5,6,8,9,10), labels=c('1NN tr','1NN te','3NN tr','3NN te', 'LDA tr', 'LDA te','NB tr','NB te'), pos=4)



#  Reconsider the above comparison, but this time ignore the stratified constraint and use a simple hold out instead when performing the splits.


  R = 49
  n = nrow(lung.data)
  ntest = .75  #training split
  
  id <- seq(1,n,by=1)
  
  ntr <- n*ntest
  nte <- n - ntr
  
  pos <- 1  #position of Y column
  
  err <- matrix(0, nrow=R, ncol = 8)  #error grid with avg tr err, avg te err for each 4 methods
  
  for (r in 1:R)
  {
    id.tr <- sample(sample(sample(id)))[1:round(ntest*n)]
    id.te <- setdiff(id, id.tr)
  
    xy.tr <- lung.data[id.tr, ]
    xy.te <- lung.data[id.te, ]
    ntr <- nrow(xy.tr)
    nte <- nrow(lung.data) - ntr
     
    yhat.1NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=1)
    err.1NN.tr <- 1-sum(diag(table(xy.tr$Y, yhat.1NN.tr)))/ntr
    yhat.1NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=1)
    err.1NN.te <- 1-sum(diag(table(xy.te$Y, yhat.1NN.te)))/nte
      
    err[r,1] <- err.1NN.tr
    err[r,2] <- err.1NN.te

    yhat.3NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=3)
    err.3NN.tr <- 1-sum(diag(table(xy.tr$Y, yhat.3NN.tr)))/ntr
    yhat.3NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=3)
    err.3NN.te <- 1-sum(diag(table(xy.te$Y, yhat.3NN.te)))/nte
      
    err[r,3] <- err.3NN.tr
    err[r,4] <- err.3NN.te
    
    lda.xy <- lda(Y~., data=xy.tr)  
    yhat.lda.tr <- predict(lda.xy, xy.tr[,-pos])$class
    yhat.lda.te <- predict(lda.xy, xy.te[,-pos])$class
    err.lda.tr <- 1-sum(diag(table(xy.tr$Y, yhat.lda.tr)))/ntr
    err.lda.te <- 1-sum(diag(table(xy.te$Y, yhat.lda.te)))/nte
      
    err[r,5] <- err.lda.tr
    err[r,6] <- err.lda.te
    
    nb.xy <- naiveBayes(Y~., data=xy.tr)
    yhat.nb.tr <- predict(nb.xy, xy.tr[,-1], type='raw')
    yhat.nb.tr <- colnames(yhat.nb.tr)[apply(yhat.nb.tr,1,which.max)]
    yhat.nb.te <- predict(nb.xy, xy.te[,-1], type='raw')
    yhat.nb.te <- colnames(yhat.nb.te)[apply(yhat.nb.te,1,which.max)]
    err.nb.tr <- 1-sum(diag(table(xy.tr$Y, yhat.nb.tr)))/ntr
    err.nb.te <- 1-sum(diag(table(xy.te$Y, yhat.nb.te)))/nte
  
    err[r,7] <- err.nb.tr
    err[r,8] <- err.nb.te
    
#    if (r%%7==0)  cat('\n', round(100*r/R,0),'completed\n')
}


#  Now we have a grid of average training and test errors for each of our splits.  Let's make a boxplot of the errors:
  

  boxplot(err[,c(2,4,6,8)], col=c(2,3,4,5), names=c('1nn','3nn','LDA','NB'),main='simple split',ylab='test errors')



#  Here they are in table form:
  

  avg.err <- c(0,0,0,0,0,0,0,0)
  for(i in 1:8)
  {
    avg.err[i] <- mean(err[,i])
  }
  avg.err.table <- matrix(0,nrow=4,ncol=2)
  count=1
  for(i in 1:4)
  {
    for(j in 1:2)
    {
      avg.err.table[i,j] <- avg.err[count]
      count <- count+1
    }
  }
  rownames(avg.err.table) <- c('1nn','3nn','lda','NB')
  colnames(avg.err.table) <- c('avg. tr. err','avg. te. err')
  avg.err.table


#  And a colorful point plot of the average test errors for good measure:
  

  plot(1:8, avg.err, ylab='Average prediction error', xlab='Method (Classifier)',xlim=c(0,9), ylim=c(0.90*min(avg.err),1.10*max(avg.err)))
  text(1:8, avg.err, col=c(2,3,4,5,6,8,9,10), labels=c('1NN tr','1NN te','3NN tr','3NN te', 'LDA tr', 'LDA te','NB tr','NB te'), pos=4)

