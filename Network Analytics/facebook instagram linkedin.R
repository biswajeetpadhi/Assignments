

###################  Network construction and visulaization   ################################################
library("igraph")


###### ???	circular network for Facebook using Adjacency Matrix from file ##################
# Load the adjacency matrix from the csv file
cir <- read.csv("E:/ASSIGNMENT/Network Analytics/Datasets_Network Analytics/facebook.csv", header = TRUE)
head(cir) 

# create a newtwork using adjacency matrix
cirNW <- graph.adjacency(as.matrix(cir), mode = "undirected")
plot(cirNW)

###### ???	star network for Instagram using Adjacency Matrix #########
star1 <- read.csv("E:/ASSIGNMENT/Network Analytics/Datasets_Network Analytics/instagram.csv", header = TRUE)
head(star1) 
# shows initial few rows of the loaded file

# create a newtwork using adjacency matrix
starNW1 <- graph.adjacency(as.matrix(star1), mode = "undirected", weighted = TRUE)
plot(starNW1)

###### ???	star network for LinkedIn using Adjacency Matrix #########
star2 <- read.csv("E:/ASSIGNMENT/Network Analytics/Datasets_Network Analytics/linkedin.csv", header = TRUE)
head(star2) 
# shows initial few rows of the loaded file

# create a newtwork using adjacency matrix
starNW2 <- graph.adjacency(as.matrix(star2), mode = "undirected", weighted = TRUE)
plot(starNW2)



