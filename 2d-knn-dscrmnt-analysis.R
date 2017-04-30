
# first generate the data-set
# with total number of points = n
# two gaussian distributions, corresponding to class y=+/-1
#   (call these M1, P1 for minus1 and plus 1)
# and Pr[Y=-1] = psi, so psi*n is number in the M1 gaussian   

library(class)
library(ElemStatLearn)
library(kernlab)
library(MASS)

n <- 300
psi <- 2.0/5

ntest <- 2.0/3

nM1 <- n*psi
nP1 <- n - nM1

muM1 <- c(-2,-1)
muP1 <- c(0,1)

sigma <- matrix(c(1,-3/4,-3/4,2),2,2)

set.seed(123)

xdataM1 <- mvrnorm(nM1, muM1, sigma)
ydataM1 <- matrix(rep.int(-1,nM1),nM1,1)

xdataP1 <- mvrnorm(nP1, muP1, sigma)
ydataP1 <- matrix(rep.int(1,nP1),nP1,1)

xdata <- rbind(xdataM1,xdataP1)
ydata <- rbind(ydataM1,ydataP1)

xydata <- data.frame(xdata,ydata)

xy <- xydata

p   <- ncol(xy)-1
pos <- p+1  
y   <- xy[,pos]  
n   <- nrow(xy)
colnames(xy)[pos] <- 'y'

#pos <- -1

# now need to randomize the data into 200 non-test and 100 test data
#
#rxydata <- xydata[,sample(ncol(xydata))]
#
#k <- 1

#  Neighborhood size 

neighborhood <-function(xy)
{
  p  <- ncol(xy)-1
  y  <- xy[,p]
  n  <- nrow(xy)
  max.k <- 10
  err.k <- matrix(0, ncol=max.k, nrow=50)
  
  for(j in 1:50)
  {
    for(k in 1:max.k)
    {
      id.tr <- sample(1:n, round(ntest*n))
      
      yhat.te <- knn(xy[id.tr,-pos],xy[-id.tr,-pos], matrix(xy[id.tr,pos]), k=k)
      err.k[j,k] <- sum(diag(prop.table(table(xy[-id.tr,p+1],yhat.te))))
    }
  }
  merr.k <- apply(err.k, 2, mean)
  return(min(which(merr.k==min(merr.k))))
} 

#  Determine the optimal neighborhood size k.opt by re-sampling

k.opt <- neighborhood(xy)

#  Set the total number of replications 

R <- 300

#  Initialize the test error vector
#     kNN, LDA, QDA

err <- matrix(0, ncol=3, nrow=R)

for(r in 1:R)
{
  
  id.F    <- which(xy$y == -1)
  n.F     <- length(id.F)
  id.F.tr <- sample(sample(sample(id.F)))[1:round(ntest*n.F)]
  id.F.te <- setdiff(id.F, id.F.tr)
  
  id.S <- which(xy$y == 1)
  n.S  <- length(id.S)
  id.S.tr <- sample(sample(sample(id.S)))[1:round(ntest*n.S)]
  id.S.te <- setdiff(id.S, id.S.tr)
  
  xy.tr <- xy[c(id.F.tr,id.S.tr), ]
  xy.te <- xy[c(id.F.te,id.S.te), ]
  ntr <- nrow(xy.tr)
  nte <- n - ntr
  
  yhat.kNN <- knn(xy.tr[,-pos], xy.te[,-pos], matrix(xy.tr[,pos]), k=k.opt)
  err.knn <- 1-sum(diag(table(xy.te$y, yhat.kNN)))/nte
  
  err[r,1] <- err.knn 
  
  lda.xy <- lda(y~., data=xy.tr)  
#  yhat.lda <- predict(lda.xy, matrix(xy.te[,1]))$class
  yhat.lda <- predict(lda.xy, xy.te[,-pos])$class
#  yhat.lda <- predict(lda.xy, xy.te)$class
  err.lda <- 1-sum(diag(table(xy.te$y, yhat.lda)))/nte
  
  err[r,2] <- err.lda
  
  qda.xy <- qda(y~., data=xy.tr)  
#  yhat.qda <- predict(qda.xy, matrix(xy.te[,1]))$class
  yhat.qda <- predict(qda.xy, xy.te[,-pos])$class
#  yhat.qda <- predict(qda.xy, xy.te)$class
  err.qda <- 1-sum(diag(table(xy.te$y, yhat.qda)))/nte
  
  err[r,3] <- err.qda
  
#  if (r%%25==0)  cat('\n', round(100*r/R,0),'completed\n')
}

bayes.risk <- 0.0384

#   windows()
quartz()
boxplot(err, col=c(2,3,4), names=c('kNN', 'LDA','QDA'))
abline(h=bayes.risk, lwd=3, col='red')



#   windows()
quartz()
avg.err <- round(apply(err, 2, mean),4)
plot(1:3, avg.err, ylab='Average prediction error', xlab='Method (Classifier)',xlim=c(0,5), ylim=c(0.90*min(avg.err),1.10*max(avg.err)))
text(1:3, avg.err, col=c(2,3,4), labels=c('kNN', 'LDA','QDA'), pos=4)
#abline(h=bayes.risk, lwd=3, col='red')


# draw decision boundary 
quartz()
GS <- 500
nd.x1 <- seq(from = min(xy[,1]), to = max(xy[,1]), length.out=GS)
nd.x2 <- seq(from = min(xy[,2]), to = max(xy[,2]), length.out=GS)
nd <- expand.grid(x = nd.x1, y = nd.x2)

prd <- as.numeric(predict(lda.xy, nd)$class)

eqscplot(xy[,1],xy[,2],col=ifelse(xy$y==-1, "red", "blue"))
#contour(x1, x2, matrix(yhat.lda,GS,GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
contour(x = nd.x1, y = nd.x2, z=matrix(prd,nrow=GS,ncol=GS), levels=c(1,2), add=TRUE, drawlabels=FALSE, col="red")
