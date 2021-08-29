# Data set genero.txt
data = read.table("genero.txt", header=TRUE, sep=",")
data = subset(data, select=c(2,3))

resumen = summary(data)
write.table(resumen, "generoResumen.txt")

png(file="generoBoxplot.png")
boxplot(data)
dev.off()

png(file="generoPlot.png")
plot(data)
dev.off()

# Data set mtcars.txt
data = read.table("mtcars.txt", header=TRUE)
data = subset(data, select=c(4,5,7))

resumen = summary(data)
write.table(resumen, "mtcarsResumen.txt")

png(file="mtcarsBoxplot.png")
boxplot(data)
dev.off()

png(file="mtcarsPlot.png")
plot(data)
dev.off()
