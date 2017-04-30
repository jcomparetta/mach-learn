

#*Here we consider the lung cancer data in file* lung-cancer-1.csv *and perform a series of learning tasks.*  

#  Perform agglomarative hierarchical clustering of the data, using either the Euclidean or Manhattan distance, and for each distance use the follwing linkages (**complete**,**average**,**ward.D**,**ward.D2**).  For each each scenario: a) generate the corresponding dendogram, b) obtain the 4 clusters cut with **cutree** and store the labels, c) partiton the dendogram into 4 rectangles using **rect.hclust**, d) visually inspect the level of match between the clustering and the true labels, and e) numerically compute a confustion matrix after properly relabelling to the hclust lables.


  lung.data <- read.csv('lung-cancer-1.csv')

 
#  There are *four* types of cancers in this dataset, corresponding to classes *k=0,1,2,3*.  
  
#  Let's standardize the data first, as we will be computing distances and we want the distances to not be dominated by measures that are larger than others.
  

     standard <-function(xx)
   {
     n <- nrow(xx)
     p <- ncol(xx)
     aa  <- matrix(rep(apply(xx,2,mean), n), ncol=p, byrow=TRUE)
     bb  <- sqrt(matrix(rep(apply(xx,2,var), n), ncol=p, byrow=TRUE))
     return((xx-aa)/bb)
     }

     lung.data[,1] <- standard(lung.data[,-1])

  
#  So we want to perform the analysis *eight times* (once for each of the four linkages, times two for the two distances).  Let's start by computing the two distance matrices, one for each of the distance types will be using (L1 norm and L2 norm).
  

      dist.L2 <- dist(lung.data[,-1], method = "euclidean", diag = FALSE, upper = FALSE)
      dist.L1 <- dist(lung.data[,-1], method = "manhattan", diag = FALSE, upper = FALSE)  


#  Now we have the two distance matrices, we can compute the cluster trees for each of the linkages by calling *hclust()* with the particular method:


      hc.L2.com <- hclust(dist.L2, method ="complete")
      hc.L2.avg <- hclust(dist.L2, method ="average")
      hc.L2.wD <- hclust(dist.L2, method ="ward.D")
      hc.L2.wD2 <- hclust(dist.L2, method ="ward.D2")
      
      hc.L1.com <- hclust(dist.L1, method ="complete")
      hc.L1.avg <- hclust(dist.L1, method ="average")
      hc.L1.wD <- hclust(dist.L1, method ="ward.D")
      hc.L1.wD2 <- hclust(dist.L1, method ="ward.D2")


#  Each hc.LX.XX holds the tree.  We could plot the dendrogram now, but to conserve trees (actual trees) we will wait until we want to partition them with rectangles.  Before we do that, let's store the labels of each of the tree with a four cluster cut.
  

      cut.L2.com <- cutree(hc.L2.com, k=4)
      cut.L2.avg <- cutree(hc.L2.avg, k=4)
      cut.L2.wD <- cutree(hc.L2.wD, k=4)
      cut.L2.wD2 <- cutree(hc.L2.wD2, k=4)
      
      cut.L1.com <- cutree(hc.L1.com, k=4)
      cut.L1.avg <- cutree(hc.L1.avg, k=4)
      cut.L1.wD <- cutree(hc.L1.wD, k=4)
      cut.L1.wD2 <- cutree(hc.L1.wD2, k=4)


#  Now the labels from the clustering are stored in each cut.LX.XX.  And now we will make our eight dendrograms, with *rect.clust()* to produce boxes to cut each tree into four clusters:
  

      plot(hc.L2.com, main = "Euclidean with complete link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L2.com, k=4)
      
      plot(hc.L2.avg, main = "Euclidean with average link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L2.avg, k=4)
      
      plot(hc.L2.wD, main = "Euclidean with ward.D link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L2.wD, k=4)
      
      plot(hc.L2.wD2, main = "Euclidean with ward.D2 link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L2.wD2, k=4)
      
      plot(hc.L1.com, main = "Manhattan with complete link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L1.com, k=4)
      
      plot(hc.L1.avg, main = "Manhattan with average link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L1.avg, k=4)
      
      plot(hc.L1.wD, main = "Manhattan with ward.D link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L1.wD, k=4)
      
      plot(hc.L1.wD2, main = "Manhattan with ward.D2 link.",labels=lung.data$Y, cex=0.4)
      rect.hclust(hc.L1.wD2, k=4)


#  Lastly lets compute a confusion matrix for each of the eight distance/method combinations.  Since we are not in a binary class situation, our confusion matrix will be a *k*x*k* matrix with 0,1,2,3 true classes across the top and 0,1,2,3 learned classes down the rows.  We can also compute the accuracy for each by summing the diagonal (summing up the true positives for each class) and then dividing by the total observations.
  

  table.L2.com <- table(lung.data$Y,cut.L2.com)
  table.L2.avg <- table(lung.data$Y,cut.L2.avg)
  table.L2.wD  <- table(lung.data$Y,cut.L2.wD)
  table.L2.wD2 <- table(lung.data$Y,cut.L2.wD2)
  
  table.L1.com <- table(lung.data$Y,cut.L1.com)
  table.L1.avg <- table(lung.data$Y,cut.L1.avg)
  table.L1.wD  <- table(lung.data$Y,cut.L1.wD)
  table.L1.wD2 <- table(lung.data$Y,cut.L1.wD2)
  
  table.L2.com
  table.L2.avg
  table.L2.wD
  table.L2.wD2
  table.L1.com
  table.L1.avg
  table.L1.wD
  table.L1.wD2


#  Some of these need some work, $cutree$ didn't get the labels correct (but that is to be expected, it doesn't know any better).
  
  library(Kmisc)
  cut.L2.com <- swap(cut.L2.com, c(1,2), c(2,1))
  cut.L2.com <- swap(cut.L2.com, c(2,4), c(4,2))
  table.L2.com <- table(lung.data$Y,cut.L2.com)

  cut.L2.avg <- swap(cut.L2.avg, c(1,2), c(2,1))
  cut.L2.avg <- swap(cut.L2.avg, c(2,3), c(3,2))
  table.L2.avg <- table(lung.data$Y,cut.L2.avg)
  
  cut.L2.wD <- swap(cut.L2.wD, c(1,2), c(2,1))
  cut.L2.wD <- swap(cut.L2.wD, c(2,3), c(3,2))
  table.L2.wD <- table(lung.data$Y,cut.L2.wD)
  
  cut.L2.wD2 <- swap(cut.L2.wD2, c(1,2), c(2,1))
  cut.L2.wD2 <- swap(cut.L2.wD2, c(2,3), c(3,2))
  table.L2.wD2 <- table(lung.data$Y,cut.L2.wD2)
  
  cut.L1.com <- swap(cut.L1.com, c(2,3), c(3,2))
  table.L1.com <- table(lung.data$Y,cut.L1.com)
  
  cut.L1.avg <- swap(cut.L1.avg, c(2,3), c(3,2))
  table.L1.avg <- table(lung.data$Y,cut.L1.avg)
  
  cut.L1.wD <- swap(cut.L1.wD, c(2,3), c(3,2))
  table.L1.wD <- table(lung.data$Y,cut.L1.wD)
  
  cut.L1.wD2 <- swap(cut.L1.wD2, c(2,3), c(3,2))
  table.L1.wD2 <- table(lung.data$Y,cut.L1.wD2)
  
#  table.L2.com
#  table.L2.avg
#  table.L2.wD
#  table.L2.wD2
#  table.L1.com
#  table.L1.avg
#  table.L1.wD
#  table.L1.wD2


#  Is there any particular scenario of hierarchical clustering that emerges as superior in the recovery of the original labels?

#  Let's look at a table of the accuracies computed by the confusion matrices.  
  
  
  n = nrow(lung.data)

  acc.L2.com <- sum(diag(table.L2.com))
  acc.L2.avg <- sum(diag(table.L2.avg))
  acc.L2.wD  <- sum(diag(table.L2.wD))
  acc.L2.wD2 <- sum(diag(table.L2.wD2))
  
  acc.L1.com <- sum(diag(table.L1.com))
  acc.L1.avg <- sum(diag(table.L1.avg))
  acc.L1.wD  <- sum(diag(table.L1.wD))
  acc.L1.wD2 <- sum(diag(table.L1.wD2))
  
  acc.table <- matrix(0,nrow=4,ncol=2)

  acc.table[1,1] <- acc.L2.com/n  
  acc.table[2,1] <- acc.L2.avg/n
  acc.table[3,1] <- acc.L2.wD/n
  acc.table[4,1] <- acc.L2.wD2/n  
  acc.table[1,2] <- acc.L1.com/n  
  acc.table[2,2] <- acc.L1.avg/n  
  acc.table[3,2] <- acc.L1.wD/n  
  acc.table[4,2] <- acc.L1.wD2/n  
  
  rownames(acc.table) <- c('complete','average','ward.D','ward.D2')
  colnames(acc.table) <- c('eucl. acc.','manh. acc.')
  acc.table


#  Looking at the results we see that the Manhattan distance with type 'complete' stands out as the most accurate predictor with 89.8% accuracy.  Overall the Manhattan distance does better than the euclidean distance.
  
##  Now let's move on to **kMeans** clustering on this same data matrix.

#  First we will attempt to determine the appropriate number of clusters in the data.  Let's look at the (total) within cluster sum of squares (WCSS) for a number of different clusters.  Since we know ahead of time that there are four labels in the dataset, we can expect an appropriate number of clusters will be around that number.  So let's try K=(1,2,...,7).
  

  nK = 7
  
  kmeans.K1 <- kmeans(lung.data[,-1],1)
  kmeans.K2 <- kmeans(lung.data[,-1],2)
  kmeans.K3 <- kmeans(lung.data[,-1],3)
  kmeans.K4 <- kmeans(lung.data[,-1],4)
  kmeans.K5 <- kmeans(lung.data[,-1],5)
  kmeans.K6 <- kmeans(lung.data[,-1],6)
  kmeans.K7 <- kmeans(lung.data[,-1],7)

#  WCSS <- matrix(c(1,0,2,0,3,0,4,0,5,0,6,0,7,0),nrow=nK,ncol=2)
  WCSS <- numeric(nK)
  WCSS[1] <- kmeans.K1$tot.withinss
  WCSS[2] <- kmeans.K2$tot.withinss
  WCSS[3] <- kmeans.K3$tot.withinss
  WCSS[4] <- kmeans.K4$tot.withinss
  WCSS[5] <- kmeans.K5$tot.withinss
  WCSS[6] <- kmeans.K6$tot.withinss
  WCSS[7] <- kmeans.K7$tot.withinss


#  Here is a plot of the WCSS versus the number of clusters.
  

  plot(1:nK,WCSS,xlab="k",ylab="total WCSS",type='o')


#  As we can see from the plot, there is an 'elbow' in the plot at K=4.  This pleases us, as it makes it easy to compare to the original labels!
  
#  (By the way, kMeans is FAST! wow!)
  
#  Let's extract the labels, and make a table of their frequencies.

  clust.K4 <- kmeans.K4$cluster
  table(clust.K4)


#  It actually looks like kMeans did a good job with the labeling, as the labels seem to be in the correct order.
  
#  Now let us compute a confusion matrix to see how well the Kmeans clustering algorithm performed.
  

  kmeans.conf.table <- table(lung.data$Y,clust.K4)
  colnames(kmeans.conf.table) <- c('0','1','2','3')
  kmeans.conf.table


#  How does the **kMeans** confusion matrix compare to the best confusion matrix from **hclust**?

#  From the kMeans confusion matrix we can calculate the percent accuracy of the clustering.  (Let's also recall the Manhattan with complete linkage confusion table from above.)
  

  kmeans.acc <- sum(diag(table(lung.data$Y,clust.K4)))/nrow(lung.data)
  kmeans.acc
  
  table.L1.com
  acc.table[1,2]
