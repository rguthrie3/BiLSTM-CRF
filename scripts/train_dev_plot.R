args <- commandArgs(trailingOnly=TRUE)
data <- read.csv(args[1])

X11()
plot(1:nrow(data), data$Train.cost, type='l', main="Train and Dev Cost",
     xlab="Epoch", ylab="Cost", col="red")
lines(1:nrow(data), data$Dev.cost, col="blue")
legend('topright', c("Train Cost", "Dev Cost"), col=c("red", "blue"), pch=1)

message("Press Return to Continue")
invisible(readLines("stdin", n=1))
