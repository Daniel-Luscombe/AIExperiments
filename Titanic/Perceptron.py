import pandas as pd
import numpy as np

def start_perceptron(X, y, w, alpha, max_iter):
    run = True
    count = 0
    updates = 0
    iteration = 0
    total = len(y)
    
    while run:
        if count >= total:
            count = 0
            if updates == 0:
                run = False
                print(f"Algorithm converged, final w = {w}")
                print(f"Total number of iterations = {iteration}")
                break
            else:
                updates = 0
                
        x_count = X[count]
        yhat = 1 if np.dot(w, x_count) >= 0 else 0
                
        if y[count] < yhat:
            w = w - alpha * x_count
            count += 1
            updates += 1
            iteration += 1
        elif y[count] > yhat:
            w = w + alpha * x_count
            count += 1
            updates += 1
            iteration += 1
        else:
            count += 1
            iteration += 1      
        
        if iteration == max_iter:
            run = False
            print(f"Algorithm did not converge, final w = {w}")
            print(f"Total number of iterations = {iteration}")
            break
            
    return w
            