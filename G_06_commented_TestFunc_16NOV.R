#                         EXTENDED STATISTICAL PROGRAMMING 
#                     PRACTICAL 4: A STOCHASTIC GRADIENT DESCENT
#                     A SIMPLE NEURAL NETWORK FOR CLASSIFICATION
# -----------------------------------------------------------------------------                                
# GROUP 6: 
# -----------------------------------------------------------------------------
# Michal Aleksandrowicz, s2614420@ed.ac.uk 
# Alexandros Stamatiou, s2614411@ed.ac.uk
# Yuli Wahyuningsih, s2465841@ed.ac.uk
# https://github.com/VanMichel/Group6.4
#-----------------------------------------------------------------------------
# MEMBER CONTRIBUTIONS: 
# -----------------------------------------------------------------------------
# Michal Aleksandrowicz: 1/3 
# Alexandros Stamatiou: 1/3
# Yuli Wahyuningsih: 1/3
# The group members feel that the workload was shared approximately fair. 
# Michal did the initial code and built the nn structure. 
# Alexandros expand and continued to build a trained network. 
# Yuli tested the data to get final results and improved the functional of code
# All code was cross-checked and commented by all members in collaboration. 
# -----------------------------------------------------------------------------
# OUTLINE:
# -----------------------------------------------------------------------------
# R functions for creating and training a simple neural 
# network (nn) for classification, using Stochastic Gradient Descent. This 
# optimization approach minimizes a loss function, while retaining good 
# generalizing power for application to other datasets. 
# -----------------------------------------------------------------------------
# HIGH-LEVEL DESCRIPTION OF CODE:
# -----------------------------------------------------------------------------
# The function netup() creates the node structure of the network according 
# to a vector "d" containing the length of nodes for each layer. The value 
# of weights "W" and biases "b" are initiated using random U(0,0.2) deviates. 
# 
# The function forward() accepts input (predictor) data and calculates 
# the corresponding node values of the network nn created with the netup() 
# function. Each layer is a linear combination of the previous one using the 
# weights and biases, transformed with the non-linear ReLu function, which 
# replaces negative node values by zeros. The node values in the last layer
# last layer are the output data (response data) and they are transformed into
# probabilities using the softmax() function. 
# 
# The dataset to be classified is assumed to be in the form (x_i, k_i) i=1,..n, 
# where the x_i (vectors) contain the input values and the k_i the corresponding
# classes. To get the network to predict the classes, the parameters W and b 
# are adjusted to minimize the loss function:
#           
#                   L = - sum_{i=1}^n log (pk_i) / n
#
# Stochastic Gradient Descent repeatedly finds the gradient of L w.r.t. the 
# parameters W and b for small randomly chosen subsets of the training data, 
# and adjust the parameters by taking a step in the direction of the negative 
# gradient. This algorithm is contained in the train() function, which uses 
# the backward() function to calculate the gradient. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START OF CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

netup <- function(d){
  ## Create a neural network structure with random weights and biases.
  ##
  ## Parameters:
  ## d: A vector representing the number of nodes in each layer of the network.
  ##
  ## Returns:
  ## A list containing the neural network structure with the following elements:
  ## - "h": A list of node values for each layer.
  ## - "W": A list of weight matrices connecting the layers.
  ## - "b": A list of bias vectors for each layer.
  
  #create h list, append values - node values
  h <- list()
  for(i in 1:length(d)){
    new_vec <- rep(NA,d[i])
    h[[length(h) + 1]] <- new_vec
  }
  
  #create W - weights matrices
  W <- list()
  for(i in 1:(length(d)-1)){
    area <- d[i]* d[i+1]
    W[[length(W) + 1]] <- matrix(runif(area,0,0.2), nrow = d[i+1],ncol = d[i])
  }
  
  #create b - biases vectors
  b <- list()
  for(i in 1:(length(d)-1)){
    b[[length(b) + 1]] <- c(runif(d[i+1],0,0.2))
  }
  
  network_list <- list("h" = h, "W" = W, "b" = b)
  return(network_list)
}

softmax <- function(vec_out){
  ## Apply softmax function to the output vector.
  ##
  ## Parameters:
  ## vec_out: Vector representing the output values.
  ##
  ## Returns:
  ## A vector of probabilities obtained by applying the softmax function.
  
  return(exp(vec_out)/sum(exp(vec_out)))
}

loss <- function(nn,inp,k){
  ## Calculate the loss for the neural network predictions.
  ##
  ## Parameters:
  ## nn: Neural network structure obtained from the netup function.
  ## inp: Matrix with input data in its rows.
  ## k: Vector of classes corresponding to each input row.
  ##
  ## Returns:
  ## The average loss.
  
  #inp is a matrix with input data in its rows; classes in vector k
  copy_network <- nn
  
  npoints <- length(inp[,1])
  #compute the summation of the -log(p_ki) terms
  sum <- 0 
  for (i in 1:npoints){
    #output for row i of inp data matrix; softmax already applied to last layer
    output <- tail(forward(copy_network,inp[i,])$h,1)[[1]] 
    #compute loss term using node k[i] 
    sum <- sum - log(output[k[i]])
  }
  
  return(sum/npoints)
}

forward <- function(nn,inp){
  ## Perform forward propagation through the neural network.
  ##
  ## Parameters:
  ## nn: Neural network structure obtained from the netup function.
  ## inp: Vector representing the input data.
  ##
  ## Returns:
  ## Updated neural network structure after forward propagation.
  
  copy_network <- nn
  
  copy_network$h[[1]] <- inp
  for(i in 1:(length(copy_network$h)-1)){
    
    #apply relu to hidden layers only
    if(i!=(length(copy_network$h)-1)){
      copy_network$h[[i+1]] <- pmax(copy_network$W[[i]]%*%copy_network$h[[i]]+copy_network$b[[i]],0)
    }else{ #and softmax to the output layer
      copy_network$h[[i+1]] <- softmax(copy_network$W[[i]]%*%copy_network$h[[i]]+copy_network$b[[i]])
    }

  }
  
  return(copy_network)
}

backward <- function(nn,k){
  ## Perform backward propagation through the neural network.
  ##
  ## Parameters:
  ## nn: Neural network structure obtained from the netup function.
  ## k: Integer representing the true class of the input.
  ##
  ## Returns:
  ## List containing the updated neural network structure and derivatives.
  
  #if k is an integer with the class number: 
  loss_deriv <- tail(nn$h,1)[[1]]
  loss_deriv[k] <- loss_deriv[k] - 1
  
  dh <- list()
  dh[[length(nn$W)+1]] <- loss_deriv
  
  dW <- list()
  db <- list()
  
  #backpropagate the loss
  for(i in rev(seq(1:length(nn$W)))){
    
    dh[[i]] <- t(nn$W[[i]])%*%(dh[[i+1]]*((nn$h[[i+1]]>0)+0L))
    dW[[i]] <- (dh[[i+1]]*((nn$h[[i+1]]>0)+0L))%*%t(nn$h[[i]])
    db[[i]] <- (dh[[i+1]]*((nn$h[[i+1]]>0)+0L))
  }
  
  #add derivatives to the results list
  derivatives <- list("dh" = dh, "dW" = dW, "db" = db)
  return(c(nn, derivatives))
}

train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  ## Train the neural network using stochastic gradient descent.
  ##
  ## Parameters:
  ## nn: Neural network structure obtained from the netup function.
  ## inp: Matrix whose rows are the datapoints.
  ## k: Corresponding vector of classes.
  ## eta: Learning rate.
  ## mb: Mini-batch size.
  ## nstep: Number of training steps.
  ##
  ## Returns:
  ## The trained neural network structure.
  
  #inp is a matrix whose rows are the datapoints
  #k is the corresponding vector of classes 1,2,3,..
  copy_network <- nn
  
  #lists for the gradients
  dLdW <- list()
  dLdb <- list()
  
  #initialize to zero 
  for(i in 1:length(copy_network$W)){
    dLdW[[i]] <- 0
    dLdb[[i]] <- 0
  }
  
  for (n in 1:nstep) {
    
    #sample mb datapoint for gradient calculation
    isample <- sample(1:length(inp[,1]), mb)
    
    for (i in isample){
      xdat <- inp[i,]
      kdat <- k[i]
      
      #compute node values
      copy_network <- forward(copy_network,xdat)
      #run backward to compute the gradients
      copy_network_grads <- backward(copy_network, kdat)
      
      #sum of gradients up to current i
      for(j in 1:length(copy_network$W)){
        dLdW[[j]] <- dLdW[[j]] + copy_network_grads$dW[[j]]
        dLdb[[j]] <- dLdb[[j]] + copy_network_grads$db[[j]]
      }
    }
    
    #update W and b in copy_network using the average of the mb gradients
    for(j in 1:length(copy_network$W)){
      copy_network$W[[j]] <- copy_network$W[[j]] - eta*dLdW[[j]]/mb 
      copy_network$b[[j]] <- copy_network$b[[j]] - eta*dLdb[[j]]/mb 
    }
    
    #reset gradients to zero for new gradient calculation
    for(j in 1:length(copy_network$W)){
      dLdW[[j]] <- 0
      dLdb[[j]] <- 0
    }
    
  }
  
  #return the trained network
  return(copy_network)
}

####----------------------------Train and test--------------------------------------####

#set seed for good predictions
set.seed(2)

nn <- netup(c(4,8,7,3))
n_d = length(nn$h) #count how many layers in the neural network's structure
d_val = iris[,-5] 
d_values = data.matrix(d_val)
d_classes = as.numeric(iris[,5]) #labeling the classes to put in vector k

#prepare a set test data consists of every 5th row, starting from row 5.
ii_test <- which((1:length(d_classes))%%5==0)
test_values <- d_values[ii_test,]
test_classes <- d_classes[ii_test]
predicted <- rep(0,length(ii_test))

#prepare a set train data
train_values <- d_values[-ii_test,]
train_classes <- d_classes[-ii_test]


#loss before training
cat("Loss before training:", loss(nn,train_values,train_classes), "\n")
#training the fit network
trained_network <- train(nn,train_values,train_classes,mb=10)

cat("Loss after training:", loss(trained_network,train_values,train_classes), "\n")


#classify the test data to species according to the class predicted             +
#+ as most probable by using the trained network
for (i in 1:length(ii_test)) {
  result <- forward(trained_network,test_values[i,])$h[[n_d]]
  predicted[i] <- which(result==max(result))
}

#compute the misclassification rate
equal <- predicted == test_classes
misclass_rate = round(length(which(equal=="FALSE"))/length(predicted),2)
cat("True test labels:", test_classes, "\n")
cat("Predicted test labels:", predicted, "\n")
cat("Misclassification rate:", misclass_rate, "\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~--END OF CODE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
