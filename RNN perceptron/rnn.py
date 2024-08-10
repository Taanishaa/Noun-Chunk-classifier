import numpy as np

class RNN:
    def __init__(self,Wax,Wya,b):
         self.Wax=Wax
         self.Wya=Wya
         self.b=b
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    def perceptron(self,a_prev,x,Wax,Wya,b):
        a_curr=self.sigmoid(np.dot(x.T,Wax)+np.dot(a_prev,Wya)+b)
        return a_curr[0][0]
    def forward(self,X,Wax,Wya,b):
        output=np.zeros((1,X.shape[0]-1))
        out=0
        for i in range(X.shape[0]-1):
            k=X[i:i+2].reshape(8,1)
            out=self.perceptron(out,k,Wax,Wya,b)
            output[0,i]=out  
        output=output[0]
        return output,Wax,Wya,b
    def rnn(self,X,Wax,Wya,b):
        output=[]
        for i in range(len(X)):
            k=np.array(X[i][0])
            out,Wax,Wya,b=self.forward(k,Wax,Wya,b)
            output.append(out)

        return  output,Wax,Wya,b 
    def prediction(self,X,Wax,Wya,b):
        ans=[]
        out,Wax,Wya,b=self.rnn(X,Wax,Wya,b)
        for i in range(len(out)):
            an=[]
            yhat=(out[i])
            for j in yhat:
                if j>=.5:
                    an.append(1)
                else:
                    an.append(0)
            ans.append(an)   
        return ans  