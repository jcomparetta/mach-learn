
### Here we consider the **banana shaped** binary classification data in file* banana-shaped-data-1.csv *and perform a series of learning tasks.*  

  library(class)
  library(ElemStatLearn)
  library(kernlab)
  library(MASS)
  library(e1071)

  banana.dat <- read.csv('banana-shaped-data-1.csv')
  str(banana.dat)

  
#  We have found the class attributes, they are in the third column, and are labeled **y**.  Now we can make a nice scatterplot of the data with the points color-coded by class.
  

  eqscplot(banana.dat[,1],banana.dat[,2],col=ifelse(banana.dat$y==0, "red", "blue"), pch=ifelse(banana.dat$y==0, "o", "x"), xlab="x1", ylab="x2")


#  Yep, that is some banana shaped data.  We won't be able to fit that with a straight line separating the two classes!
  
#  Consider nu-svc and create a grid of 27 possible values for $nu$ in the interval [0,1] and use the ksvm(...,'nu-svc', cross=5, kernel='rbfdot') from library(kernlab) to obtain the most plausible value of $nu$ by 5-fold CV.  Plot the CV error as a function of $nu$, saving the best value of $nu$ for later.

#  When the data cannon be seperated by a straight line, we need to resort to kernel svm, which is when we find a mapping of the given data into a higher dimensional space where the classes can be better separated by a hyperplane.
  
#  The first kernel we will be trying is the (gaussian) Radial Basis Function with $nu$-svm.

#  I received errors when running the code for values of $nu$=0 and $nu$>.9 so I will use the bounds of the values for $nu$ for CV of (0,.9]


  library(kernlab)
  library(e1071)
  
  xy <- banana.dat
  
  pos <- 3  #y is in the third column
  colnames(xy)[pos] <- 'y'
  n <- nrow(xy)
  p = 2   #we have x1 and x2 as our variables
  
  nNu <- 27  #number of hyperparameters
  
  vNu <- seq(0,.9,length.out=nNu)
  vNu[1] <- 0.0000001
  
  cv.Nu <- matrix(0,ncol=1,nrow=nNu)
  for(j in 1:nNu)
  {
    nu.svm.xy  <- ksvm(y~., type='nu-svc', data=xy, cross=5, kernel='rbfdot', nu=vNu[j])      
    cv.Nu[j, 1]   <- cross(nu.svm.xy)
  }

#  Now our CV errors are stored in *cv.Nu* and we can make a plot to pick out the optimal value of $nu$.
  

  plot(vNu[1:nNu],cv.Nu,xlab="nu",main='CV for nu',ylab="cv error",type='o')


#  From our plot we can see that the optimal value of $nu$ is the second value in the CV loop.  We can check the actual value with some r code.:
  

  optNu <- which.min(cv.Nu[,1])
  vNu[optNu]


#  Consider C-svc and create a grid of 27 possible values for $C$ in the interval [$2^-7$,$2^7$] and use the ksvm(...,'C-svc', cross=5, kernel='rbfdot') from library(kernlab) to obtain the most plausible value of $C$ by 5-fold CV.  Plot the CV error as a function of $C$, saving the best value of $nu$ for later.
  
#  The second kernel we will be trying is the (gaussian) Radial Basis Function with C-SVM.


  nC <- 27
  vC <- 2^(seq(-7,7, length=nC))
  
  cv.C <- matrix(0,ncol=1,nrow=nC)
  for(j in 1:nC)
  {
    c.svm.xy  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='rbfdot', C=vC[j])      
    cv.C[j, 1]   <- cross(c.svm.xy)
  }
```

#  Now our CV errors are stored in *cv.C* and we can make a plot to pick out the optimal value of $C$.
  

  plot(vC[1:nC],cv.C,xlab="C",main='CV for C',ylab="cv error",type='o')


#  From our plot we can see that the optimal value of $C$ is around 3.  We can check this with some r code:
  
  optC_grbf <- which.min(cv.C[,1])
  vC[optC_grbf]


#  Consider the $C-svc$ with the same grid as above and find the optimal C with the following 5 kernels: 
#-Laplace Radial Basis Function kernel
#-Hyperbolic Tangent kernel
#-Linear kernel
#-Polynomial kernel
#-Anova kernel

#  That is a lot of kernels. Let's look at them all at once (note, some of these kernels have hyperparameters of their own.  For polynomial we will use scale=offset=1,degree=2 and for hyperbolic tangent we will use scale=offset=1 and for anova we will use sigma=0.5,degree=2.  The rest will use the default parameters):

#  nC <- 27
#  vC <- 2^(seq(-7,7, length=nC))
  
  cv.C.lrbf <- matrix(0,ncol=1,nrow=nC)
  cv.C.hyp <- matrix(0,ncol=1,nrow=nC)
  cv.C.lin <- matrix(0,ncol=1,nrow=nC)
  cv.C.pol <- matrix(0,ncol=1,nrow=nC)
  cv.C.anv <- matrix(0,ncol=1,nrow=nC)
  
  for(j in 1:nC)
  {
    c.svm.lrbf  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='laplacedot', C=vC[j])      
    cv.C.lrbf[j, 1]   <- cross(c.svm.lrbf)
    
    c.svm.hyp  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='tanhdot', kpar= list(scale=1, offset=1), C=vC[j])   
    cv.C.hyp[j, 1]   <- cross(c.svm.hyp)
    
    c.svm.lin  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='vanilladot', C=vC[j])      
    cv.C.lin[j, 1]   <- cross(c.svm.lin)
    
    c.svm.pol  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='polydot', kpar= list(degree=2, scale=1, offset=1),C=vC[j])      
    cv.C.pol[j, 1]   <- cross(c.svm.pol)

    c.svm.anv  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='anovadot', kpar=list(sigma=0.5,degree=2),C=vC[j])      
    cv.C.anv[j, 1]   <- cross(c.svm.anv)
  }
  
  CVgrid <- matrix(0,nrow=6,ncol=2)
  
  CVgrid[1,1] <- optC_grbf
  CVgrid[1,2] <- cv.C[optC_grbf,1]  
  
  optC_lrbf <- which.min(cv.C.lrbf[,1])
  CVgrid[2,1] <- vC[optC_lrbf]
  CVgrid[2,2] <- cv.C.lrbf[optC_lrbf,1]
  
  optC_hyp <- which.min(cv.C.hyp[,1])
  CVgrid[3,1] <- vC[optC_hyp]
  CVgrid[3,2] <- cv.C.hyp[optC_hyp,1]
  
  optC_lin <- which.min(cv.C.lin[,1])
  CVgrid[4,1] <- vC[optC_lin]
  CVgrid[4,2] <- cv.C.lin[optC_lin,1]
  
  optC_pol <- which.min(cv.C.pol[,1])
  CVgrid[5,1] <- vC[optC_pol]
  CVgrid[5,2] <- cv.C.pol[optC_pol,1]
  
  optC_anv <- which.min(cv.C.anv[,1])
  CVgrid[6,1] <- vC[optC_anv]
  CVgrid[6,2] <- cv.C.anv[optC_anv,1]
  
  rownames(CVgrid) <- c('grbf','lrbf','hyp','lin','poly','anova')
  colnames(CVgrid) <- c('C-opt','CVerr')
  CVgrid


#  Above are the values of the CV error at the optimal value of C (i.e. where CV error is mimimized).  
    
#    Let's also make a cross validation plot for each of the kernels.  
   
  par(mfrow=c(3,2))
  plot(vC[1:nC],cv.C,xlab="C",main='CV for C-svm (grbf)',ylab="cv error",type='o',col='1')
  plot(vC[1:nC],cv.C.lrbf,xlab="C",main='CV for C-svm (lrbf)',ylab="cv error",type='o',col='2')
  plot(vC[1:nC],cv.C.hyp,xlab="C",main='CV for C-svm (hyp tan)',ylab="cv error",type='o',col='3')
  plot(vC[1:nC],cv.C.lin,xlab="C",main='CV for C-svm (linear)',ylab="cv error",type='o',col='4')
  plot(vC[1:nC],cv.C.pol,xlab="C",main='CV for C-svm (poly)',ylab="cv error",type='o',col='5')
  plot(vC[1:nC],cv.C.anv,xlab="C",main='CV for C-svm (anova)',ylab="cv error",type='o',col='6')


#    And let's overplot all of them to better compare.  Most of the action is for C<10 so we won't show all the way to 2^7.
  

  plot(vC[1:nC],cv.C,xlab="C",main='CV for 6 methods of C-SVM',ylab="cv error",type='o',col='1',ylim=c(0,0.5),xlim=c(0,7))
  lines(vC[1:nC],cv.C.lrbf,type='o',col='2')
  lines(vC[1:nC],cv.C.hyp,type='o',col='3')
  lines(vC[1:nC],cv.C.lin,type='o',col='4')
  lines(vC[1:nC],cv.C.pol,type='o',col='5')
  lines(vC[1:nC],cv.C.anv,type='o',col='6')
  legend(5,0.4,c('grbf','lrbf','hyper','linear','polynomial','anova'),lty=c(1,1),lwd=c(2.5,2.5),col=c('1','2','3','4','5','6'),title='kernel')

  
#  Comparison of Predictive performances.  Consider four classifiers: **1NN**, **3NN**, **LDA**, **QDA**, **Naive Bayes**, **$nu$-SVC**, and **$C$-SVC**.  For each of these classifiers use a *stratified hold out* split with 75% Training and 25% test.  (Stratified hold out means that the training and test data must be consistent with the prior's found above).  Perform R=49 splits, computing the test error for each split.

#  We will use the optimal hyperparameters found above for $nu$ and $C$.
  

  r = 49
  ntest = 0.75 #training split
  ntr <- round(n*ntest)
  nte <- n-ntr
  
  pos <-3
  
  err.str <- matrix(0,nrow=r,ncol=14) #err matrix for (train & test)*7 methods
  
  for (j in 1:r)
  {
    id.zero    <- which(xy$y == 0)
    n.zero     <- length(id.zero)
    id.zero.tr <- sample(sample(sample(id.zero)))[1:round(ntest*n.zero)]
    id.zero.te <- setdiff(id.zero, id.zero.tr)
  
    id.one    <- which(xy$y == 1)
    n.one     <- length(id.one)
    id.one.tr <- sample(sample(sample(id.one)))[1:round(ntest*n.one)]
    id.one.te <- setdiff(id.one, id.one.tr)
  
    xy.tr <- xy[c(id.zero.tr,id.one.tr), ]
    xy.te <- xy[c(id.zero.te,id.one.te), ]
    ntr <- nrow(xy.tr)
    nte <- n - ntr
  
    yhat.1NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=1)
    err.1NN.tr <- 1-sum(diag(table(xy.tr$y, yhat.1NN.tr)))/ntr
    yhat.1NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=1)
    err.1NN.te <- 1-sum(diag(table(xy.te$y, yhat.1NN.te)))/nte
    
    err.str[j,1] <- err.1NN.tr
    err.str[j,2] <- err.1NN.te
      
    yhat.3NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=3)
    err.3NN.tr <- 1-sum(diag(table(xy.tr$y, yhat.3NN.tr)))/ntr
    yhat.3NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=3)
    err.3NN.te <- 1-sum(diag(table(xy.te$y, yhat.3NN.te)))/nte
      
    err.str[r,3] <- err.3NN.tr
    err.str[r,4] <- err.3NN.te
      
    lda.xy <- lda(y~., data=xy.tr)  
    yhat.lda.tr <- predict(lda.xy, xy.tr[,-pos])$class
    yhat.lda.te <- predict(lda.xy, xy.te[,-pos])$class
    err.lda.tr <- 1-sum(diag(table(xy.tr$y, yhat.lda.tr)))/ntr
    err.lda.te <- 1-sum(diag(table(xy.te$y, yhat.lda.te)))/nte
      
    err.str[r,5] <- err.lda.tr
    err.str[r,6] <- err.lda.te
      
    qda.xy <- qda(y~., data=xy.tr)  
    yhat.qda.tr <- predict(qda.xy, xy.tr[,-pos])$class
    yhat.qda.te <- predict(qda.xy, xy.te[,-pos])$class
    err.qda.tr <- 1-sum(diag(table(xy.tr$y, yhat.lda.tr)))/ntr
    err.qda.te <- 1-sum(diag(table(xy.te$y, yhat.lda.te)))/nte
      
    err.str[r,7] <- err.lda.tr
    err.str[r,8] <- err.lda.te
    
    nb.xy <- naiveBayes(y~., data=xy.tr)
    yhat.nb.tr <- predict(nb.xy, xy.tr[,-pos], type='raw')
    yhat.nb.tr <- apply(yhat.nb.tr,1,which.max)
    yhat.nb.te <- predict(nb.xy, xy.te[,-pos], type='raw')
    yhat.nb.te <- apply(yhat.nb.te,1,which.max)
    err.nb.tr <- 1-sum(diag(table(xy.tr$y, yhat.nb.tr)))/ntr
    err.nb.te <- 1-sum(diag(table(xy.te$y, yhat.nb.te)))/nte
    
    err.str[j,9] <- err.nb.tr
    err.str[j,10] <- err.nb.te
    
    nu.svm.xy  <- ksvm(y~., type='nu-svc', data=xy.tr, cross=0, kernel='rbfdot', nu=vNu[optNu])      
    yhat.nu.svm.tr <- predict(nu.svm.xy,xy.tr[,-pos])
    err.nu.svm.tr <- 1-sum(diag(table(xy.tr$y, yhat.nu.svm.tr)))/ntr      
    yhat.nu.svm.te <- predict(nu.svm.xy,xy.te[,-pos])
    err.nu.svm.te <- 1-sum(diag(table(xy.te$y, yhat.nu.svm.te)))/nte
    
    err.str[j, 11]   <- err.nu.svm.tr
    err.str[j, 12]   <- err.nu.svm.te
      
    c.svm.xy  <- ksvm(y~., type='C-svc', data=xy.tr, cross=0, kernel='rbfdot', C=vC[optC_grbf])      
    yhat.c.svm.tr <- predict(c.svm.xy,xy.tr[,-pos])
    err.c.svm.tr <- 1-sum(diag(table(xy.tr$y, yhat.c.svm.tr)))/ntr      
    yhat.c.svm.te <- predict(c.svm.xy,xy.te[,-pos])
    err.c.svm.te <- 1-sum(diag(table(xy.te$y, yhat.c.svm.te)))/nte
    
    err.str[j, 13]   <- err.c.svm.tr
    err.str[j, 14]   <- err.c.svm.te
  }


#  Now we have a grid of average training and test errors for each of our splits.  Let's make a boxplot of the errors:
  

#  boxplot(err.str[,c(2,4,6,8,10,12,14)], col=c(2,3,4,5,6,8,9), names=c('1nn','3nn','lda','qda','nb','nu-svm','c-svm'),ylab="test errors")
  boxplot(err.str[,c(2,4,6,8,12,14)], col=c(2,3,4,5,6,8,9), names=c('1nn','3nn','lda','qda','nu-svm','c-svm'),ylab="test errors")



#  Here they are in table form:
  

  avg.err.str <- numeric(14)
  for(i in 1:14)
  {
    avg.err.str[i] <- mean(err.str[,i])
  }
  avg.err.table <- matrix(0,nrow=7,ncol=2)
  count=1
  for(i in 1:7)
  {
    for(j in 1:2)
    {
      avg.err.table[i,j] <- avg.err.str[count]
      count <- count+1
    }
  }
  rownames(avg.err.table) <- c('1nn','3nn','lda','qda','NB','nu-svm','c-svm')
  colnames(avg.err.table) <- c('avg. tr. err','avg. te. err')
  avg.err.table


#  And a colorful point plot of the average test errors for good measure:
  

  plot(1:7, avg.err.table[,2], ylab='Average test error', xlab='Method (Classifier)',xlim=c(0,8), ylim=c(0.90*min(avg.err.str),1.10*max(avg.err.str)))
  text(1:7, avg.err.table[,2], col=c(2,3,4,5,6,8,9), labels=c('1NN','3NN','LDA','QDA','NB','nu-svm','c-svm'), pos=4)



#  Reconsider the above comparison, but this time with a simple hold out instead of the stratified hold out.

#  Easy enough, let's make a few modifications to the above R code to accomplish this task.
  r = 49

  id <- seq(1,n,by=1)
  
  ntr <- round(n*ntest)
  nte <- n - ntr
   
  pos  <- 3
  
  err <- matrix(0,nrow=r,ncol=14) 
  
  for (j in 1:r)
  {
    id.tr <- sample(sample(sample(id)))[1:round(ntest*n)]
    id.te <- setdiff(id, id.tr)
  
    xy.tr <- xy[id.tr, ]
    xy.te <- xy[id.te, ]
    ntr <- nrow(xy.tr)
    nte <- nrow(xy) - ntr

    yhat.1NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=1)
    err.1NN.tr <- 1-sum(diag(table(xy.tr$y, yhat.1NN.tr)))/ntr
    yhat.1NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=1)
    err.1NN.te <- 1-sum(diag(table(xy.te$y, yhat.1NN.te)))/nte
    
    err[j,1] <- err.1NN.tr
    err[j,2] <- err.1NN.te
      
    yhat.3NN.tr <- knn(xy.tr[,-pos],  xy.tr[,-pos], xy.tr[,pos], k=3)
    err.3NN.tr <- 1-sum(diag(table(xy.tr$y, yhat.3NN.tr)))/ntr
    yhat.3NN.te <- knn(xy.tr[,-pos],  xy.te[,-pos], xy.tr[,pos], k=3)
    err.3NN.te <- 1-sum(diag(table(xy.te$y, yhat.3NN.te)))/nte
      
    err[r,3] <- err.3NN.tr
    err[r,4] <- err.3NN.te
      
    lda.xy <- lda(y~., data=xy.tr)  
    yhat.lda.tr <- predict(lda.xy, xy.tr[,-pos])$class
    yhat.lda.te <- predict(lda.xy, xy.te[,-pos])$class
    err.lda.tr <- 1-sum(diag(table(xy.tr$y, yhat.lda.tr)))/ntr
    err.lda.te <- 1-sum(diag(table(xy.te$y, yhat.lda.te)))/nte
      
    err[r,5] <- err.lda.tr
    err[r,6] <- err.lda.te
      
    qda.xy <- qda(y~., data=xy.tr)  
    yhat.qda.tr <- predict(qda.xy, xy.tr[,-pos])$class
    yhat.qda.te <- predict(qda.xy, xy.te[,-pos])$class
    err.qda.tr <- 1-sum(diag(table(xy.tr$y, yhat.lda.tr)))/ntr
    err.qda.te <- 1-sum(diag(table(xy.te$y, yhat.lda.te)))/nte
      
    err[r,7] <- err.lda.tr
    err[r,8] <- err.lda.te
    
    nb.xy <- naiveBayes(y~., data=xy.tr)
    yhat.nb.tr <- predict(nb.xy, xy.tr[,-pos], type='raw')
    yhat.nb.tr <- apply(yhat.nb.tr,1,which.max)
    yhat.nb.te <- predict(nb.xy, xy.te[,-pos], type='raw')
    yhat.nb.te <- apply(yhat.nb.te,1,which.max)
    err.nb.tr <- 1-sum(diag(table(xy.tr$y, yhat.nb.tr)))/ntr
    err.nb.te <- 1-sum(diag(table(xy.te$y, yhat.nb.te)))/nte
    
    err[j,9] <- err.nb.tr
    err[j,10] <- err.nb.te
    
    nu.svm.xy  <- ksvm(y~., type='nu-svc', data=xy.tr, cross=5, kernel='rbfdot', nu=vNu[optNu])      
    yhat.nu.svm.tr <- predict(nu.svm.xy,xy.tr[,-pos])
    err.nu.svm.tr <- 1-sum(diag(table(xy.tr$y, yhat.nu.svm.tr)))/ntr      
    yhat.nu.svm.te <- predict(nu.svm.xy,xy.te[,-pos])
    err.nu.svm.te <- 1-sum(diag(table(xy.te$y, yhat.nu.svm.te)))/nte
    
    err[j, 11]   <- err.nu.svm.tr
    err[j, 12]   <- err.nu.svm.te
      
    c.svm.xy  <- ksvm(y~., type='C-svc', data=xy.tr, cross=5, kernel='rbfdot', C=vC[optC_grbf])      
    yhat.c.svm.tr <- predict(c.svm.xy,xy.tr[,-pos])
    err.c.svm.tr <- 1-sum(diag(table(xy.tr$y, yhat.c.svm.tr)))/ntr      
    yhat.c.svm.te <- predict(c.svm.xy,xy.te[,-pos])
    err.c.svm.te <- 1-sum(diag(table(xy.te$y, yhat.c.svm.te)))/nte
    
    err[j, 13]   <- err.c.svm.tr
    err[j, 14]   <- err.c.svm.te
  }

  
#  Now we have a grid of average training and test errors for each of our splits.  Let's make a boxplot of the errors:
  

  boxplot(err[,c(2,4,6,8,10,12,14)], col=c(2,3,4,5,6,8,9), names=c('1nn','3nn','lda','qda','nb','nu-svm','c-svm'),ylab="test errors")



#  Here they are in table form:
  
  avg.err.str <- numeric(14)
  for(i in 1:14)
  {
    avg.err.str[i] <- mean(err.str[,i])
  }
  avg.err.table <- matrix(0,nrow=7,ncol=2)
  count=1
  for(i in 1:7)
  {
    for(j in 1:2)
    {
      avg.err.table[i,j] <- avg.err.str[count]
      count <- count+1
    }
  }
  rownames(avg.err.table) <- c('1nn','3nn','lda','qda','NB','nu-svm','c-svm')
  colnames(avg.err.table) <- c('avg. tr. err','avg. te. err')
  avg.err.table


#  And a colorful point plot of the average test errors for good measure:
  

  plot(1:7, avg.err.table[,2], ylab='Average test error', xlab='Method (Classifier)',xlim=c(0,8), ylim=c(0.90*min(avg.err.str),1.10*max(avg.err.str)))
  text(1:7, avg.err.table[,2], col=c(2,3,4,5,6,8,9), labels=c('1NN','3NN','LDA','QDA','NB','nu-svm','c-svm'), pos=4)


#  For fun, let's check out some decision boundary plots.


GS <- 100
nd.x1 <- seq(from = min(xy[,1]), to = max(xy[,1]), length.out=GS)
nd.x2 <- seq(from = min(xy[,2]), to = max(xy[,2]), length.out=GS)
nd <- expand.grid(x = nd.x1, y = nd.x2)

par(mfrow=c(2,2))

yhat.1NN <- knn(xy[,-pos],  nd, xy[,pos], k=1)
prd.1NN <- as.numeric(yhat.1NN)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='1NN decision boundary',xlab='x1',ylab='x2')
contour(x = nd.x1, y = nd.x2, z=matrix(prd.1NN,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

yhat.3NN <- knn(xy[,-pos],  nd, xy[,pos], k=3)
prd.3NN <- as.numeric(yhat.3NN)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='3NN decision boundary',xlab='x1',ylab='x2')
contour(x = nd.x1, y = nd.x2, z=matrix(prd.3NN,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

lda.xy <- lda(y~., data=xy)
prd.lda <- as.numeric(predict(lda.xy, newdata=data.frame(X1=nd[,1], X2=nd[,2]))$class)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='lda decision boundary',xlab='x1',ylab='x2')
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd.lda,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

qda.xy <- qda(y~., data=xy)
prd.qda <- as.numeric(predict(qda.xy, newdata=data.frame(X1=nd[,1], X2=nd[,2]))$class)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='qda decision boundary',xlab='x1',ylab='x2')
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd.qda,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

par(mfrow=c(2,2))

nb.xy <- naiveBayes(y~., data=xy)
prd.nb <- predict(nb.xy, newdata=data.frame(X1=nd[,1], X2=nd[,2]), type='raw')
prd.nb <- apply(prd.nb,1,which.max)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='NB decision boundary',xlab='x1',ylab='x2')
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd.nb,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

nu.svm.xy  <- ksvm(y~., type='nu-svc', data=xy, cross=5, kernel='rbfdot', nu=vNu[optNu]) 
prd.nu.svm <- predict(nu.svm.xy, newdata=data.frame(X1=nd[,1], X2=nd[,2]), type="response")

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='nu-svm decision boundary',xlab='x1',ylab='x2')
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd.nu.svm,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")

c.svm.xy  <- ksvm(y~., type='C-svc', data=xy, cross=5, kernel='rbfdot', C=vC[optC_grbf]) 
prd.c.svm <- predict(c.svm.xy, newdata=data.frame(X1=nd[,1], X2=nd[,2]), type="response")

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==0, "red", "blue"),main='c-svm decision boundary',xlab='x1',ylab='x2')
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd.c.svm,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="black")
