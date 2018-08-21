import numpy
#import panda
#Define our functions

#This program calcaultes a two-class Gaussian Discriminant

#This function calculates prior probabilities for each of two classes
def CalculatePriors(X_trn, y_trn):
    rows,cols=y_trn.shape
    countC1=0
    countC2=0
    i=0
    end=cols

    #Count for each class
    while(i<rows):
        if y_trn[i][end-1]==0: 
            countC1=countC1+1
        if y_trn[i][end-1]==1:
            countC2=countC2+1
        i=i+1

    #Calculate probability of each class
    prior1=countC1/(countC1+countC2)
    prior2=countC2/(countC1+countC2)
    return prior1, prior2

#This function calculates the Gaussian discriminant    
def CalculateGaussianDiscr(X,m,S,prior):
    rows,cols=X.shape
    g=numpy.zeros(shape=(rows,1))

    #Calculate Gaussian discriminant for each row in the matrix    
    for i in range(0,rows):  
        q=rowVector(X,i)
        part0=-1/2*numpy.log(numpy.linalg.det(S))
        inside=numpy.matrix.transpose(q)-m
        part1=-1/2*numpy.matrix.transpose(inside)
        part2=numpy.dot(part1,numpy.linalg.inv(S))
        part3=numpy.dot(part2,numpy.matrix.transpose(q)-m)
        g[i][0]=part0+part3+numpy.log(prior)
    return g

#This function calculates the error rate of the estimates
def CalculateErrorRate(g1,g2,y_tst):
    rows,cols=y_tst.shape
    z=numpy.zeros(shape=(rows,1))
    rightcount=0

    for i in range(0,rows): 
        if g1[i][0]>g2[i][0]:
            z[i][0]=0
        else:
            z[i][0]=1

        if z[i][0]==y_tst[i][0]:
            rightcount=rightcount+1 
    error_rate=1-rightcount/rows  
    return error_rate


#This function calculates covariance
def CalculateMeanIndepCov(X,y): 
    rows,cols=X.shape
    m1=numpy.zeros(shape=(cols,1))
    m2=numpy.zeros(shape=(cols,1))
    sum1=0
    sum2=0
    count1=0
    count2=0

    #Calculate means
    for j in range(0,cols):
        sum1=0
        count1=0
        sum2=0
        count2=0

        for i in range(1,rows):  
            if y[i][0]==0:
                sum1=sum1+X[i][j]
                count1=count1+1
            if y[i][0]==1:
                sum2=sum2+X[i][j]
                count2=count2+1
  #          print(sum2)
        m1[j][0]=sum1/count1
        m2[j][0]=sum2/count2

    S1=numpy.zeros(shape=(cols,cols))
    S2=numpy.zeros(shape=(cols,cols))

    #Calculate covariance
    for k in range(1,rows):
        v=rowVector(X,k)
        v=numpy.matrix.transpose(v)
        if y[k][0]==0:
             S1=S1+numpy.dot((v-m1),numpy.matrix.transpose(v-m1))
        if y[k][0]==1:
             S2=S2+numpy.dot((v-m2),numpy.matrix.transpose(v-m2))
             
    S1=S1/count1
    S2=S2/count2
            
    return m1,m2,S1,S2     


#This function reads in data
def ReadData(training_filename, test_filename):  
    traindata=numpy.genfromtxt(training_filename, delimiter=',')
    testdata=numpy.genfromtxt(test_filename, delimiter=',')
    m,n=traindata.shape
    p,q=testdata.shape

    X_trn=numpy.transpose(traindata)
    X_trn=numpy.delete(X_trn,n-1,0)
    X_trn=numpy.transpose(X_trn)
    y_trn=colVector(traindata,n-1)

    X_tst=numpy.transpose(testdata)
    X_tst=numpy.delete(X_tst,q-1,0)
    X_tst=numpy.transpose(X_tst)
    y_tst=colVector(testdata,q-1)

    return X_trn,y_trn,X_tst,y_tst


#This function produces a column vector
def colVector(A,r):   
    m,n=A.shape
    q=numpy.zeros(shape=(m,1))
    for i in range(0,m):
        q[i][0]=A[i][r]    
    return q

#This function produces a row vector
def rowVector(A,r):  
    m,n=A.shape
    q=numpy.zeros(shape=(1,n))
    for i in range(0,n):
        q[0][i]=A[r][i]    
    return q



#The Executable Commands of the Program

#Import files and organize test and training data sets
trainingFile='train.txt' 
testFile='test.txt' 
X_trn,y_trn,X_tst,y_tst=ReadData(trainingFile, testFile)

#Calculate prior probabilities
prior1, prior2=CalculatePriors(X_trn,y_trn)

#Calculate covariance matricies and mean vector
m1,m2,S1,S2=CalculateMeanIndepCov(X_trn,y_trn)

#Calculate gaussian discriinants for class 1
g1=CalculateGaussianDiscr(X_tst, m1,S1,prior1) 

#Calculate gaussian discriminants for class 2
g2=CalculateGaussianDiscr(X_tst, m2,S2,prior2)

#Calculate and print error rate
error_rate=CalculateErrorRate(g1,g2,y_tst)
print(error_rate)





