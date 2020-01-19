import numpy as np
from scipy.stats import rankdata
from tabulate import tabulate

def main():
    import sys
    import pandas as pd
    if len(sys.argv)!=4:
        print("ERROR! WRONG NUMBER OF PARAMETERS")
        print("USAGES: $python <programName> <dataset> <weights array> <impacts array>")
        print("EXAMPLE: $python programName.py data.csv '1,1,1,1' '+,+,-,+' ")
        exit(1)

    dataset = pd.read_csv(sys.argv[1]).values             #importing the dataset
    decisionMatrix = dataset[:,1:]                        #dropping first column
    weights = [int(i) for i in sys.argv[2].split(',')]    #initalizing weights array
    impacts = sys.argv[3].split(',')                      #initalizing impacts array
    topsis(decisionMatrix , weights , impacts)
    
def topsis(decisionMatrix,weights,impacts):
    r,c = decisionMatrix.shape
    if len(weights) != c :
        return print("ERROR! length of 'weights' is not equal to number of columns")
    if len(impacts) != c :
        return print("ERROR! length of 'impacts' is not equal to number of columns")
    if not all(i > 0 for i in weights) :
        return print("ERROR! weights must be positive numbers")
    if not all(i=="+"or i=="-" for i in impacts) :
        return print("ERROR! impacts must be a character vector of '+' and '-' signs")

    data = np.zeros([r+2,c+4])
    
    
    for i in range(r):
        for j in range(c):
            data[i,j] = (decisionMatrix[i,j]/np.sqrt(sum(decisionMatrix[:,j]**2)))*weights[j]
    
    for i in range(c):
        data[r,i] = max(data[:r,i])
        data[r+1,i] = min(data[:r,i])
        if impacts[i] == "-":
            data[r,i] , data[r+1,i] = data[r+1,i] , data[r,i]
    
    for i in range(r):
        data[i,c] = np.sqrt(sum((data[r,:c] - data[i,:c])**2))
        data[i,c+1] = np.sqrt(sum((data[r+1,:c] - data[i,:c])**2))
        data[i,c+2] = data[i,c+1]/(data[i,c] + data[i,c+1])        #position 
        
    #data[:r,c+3] = len(data[:r,c+2]) - rankdata(data[:r,c+2]).astype(int) + 1
    data[:r,c+3]=rankdata(data[:r,c+2],method='min')
    print(tabulate({"Model": np.arange(1,r+1), "Score": data[:5,c+2], "Rank": data[:5,c+3]}, headers="keys"))
    
if __name__ == "__main__":
    main()