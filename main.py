import pandas as pd # Library for investigating the dataset
import numpy as np # Library for investigating the dataset
import matplotlib.pyplot as plt # Library for visualising the dataset and resuklt
import copy # Library to copy something
import math # Library for some math operations

# Main function
def main():
    exp, salaries = read_csv_file() # Reading csv file
    visuale_init_data(exp, salaries) # Visualising the initial dataset

    init_w = 0. # Initial w
    init_b = 0. # Initial b
    iterations = 70 # Number of iterations
    alpha = 0.0001 # Learning rate

    w, b ,_ , _ = gradient_descent(exp, salaries, init_w, init_b, compute_cost, compute_gradient, alpha, iterations) # Our final w and b values found by gradient descent
    print("w,b found by gradient descent:", w, b) # Printing the new w and b parameters

    visuale_model(exp, salaries, w, b) # Visualising our model

    predict_inp = int(input("Give the working experience in months : ")) # Predict input (Taken from the user)
    prediction = predict(w, b, predict_inp) # Make the prediction
    print("{} months working experience salary prediction is : {}".format(predict_inp, prediction)) # Print it on screen

# Reading the csv file
def read_csv_file():
    df = pd.read_csv("exp_salary.csv") # Read csv file

    exp = list(df["exp(in months)"][:100]) # Take the first 100 data from the exp column
    salary = list(df["salary(in thousands)"][:100]) # Take the first 100 data from the salary column

    exp = np.array(exp) # Convert it to a np array
    salary = np.array(salary) # Convert it to a np array

    return exp, salary

def visuale_init_data(exp, salaries): # Visualises initial dataset
    plt.scatter(exp, salaries, marker="x", c="red") # Scatter plot
    plt.title("Experience - Salaries") # Graph's title
    plt.xlabel("Experience") # Name of the x axes
    plt.ylabel("Salaries") # Name of the y axes
    plt.show() # Show the graph

def compute_cost(x_train, y_train, w , b): # Computes the cost
    total_cost = 0 # Initial cost variable
    m = x_train.shape[0] # Number of examples in the dataset
    cost = 0 # Cost variable
    for i in range(m): # For each example in the dataset calculate the cost
        f_wb = w * x_train[i] + b
        cost += (f_wb - y_train[i]) ** 2
    
    total_cost = (1 / (2 * m)) * cost # Find the final total cost

    return total_cost

def compute_gradient(x_train, y_train, w ,b): # Computes gradient
    m = x_train.shape[0] #Number of examples in the dataset

    dj_dw = 0 # The partial derivative of J with respect to w
    dj_db = 0 # The partial derivative of J with respect to b

    # Find the final derivatives by using each data in dataset
    for i in range(m):
        f_wb = w * x_train[i] + b
        dj_dw_i = (f_wb - y_train[i]) * x_train[i]
        dj_db_i = (f_wb - y_train[i]) * 1

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x_train, y_train, w_init, b_init, cost_function, gradient_function, alpha, num_iters): # Find the best w and b values using gradient descent
    J_history = [] # To hold the data
    w_history = [] # To hold the w values
    w = copy.deepcopy(w_init)  # We copied it not to override the initial w
    b = b_init # Our initial b value

    for i in range(num_iters): # Execute gradient descent
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x_train, y_train, w, b)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    # Return w, b values
    return w, b, J_history, w_history

def visuale_model(exp, salaries, w, b): # Visualises the model
    m = len(exp) # Number of examples in the dataset
    predicted = np.zeros(m) # To hold the predicted data

    for i in range(m): # Fill the list with predicted data using new w and b values
        predicted[i] = w * exp[i] + b

    # Plot the data and the model.
    plt.plot(exp, predicted, c = "b")
    plt.scatter(exp, salaries, marker='x', c='r') 

    plt.title("Experiences vs. Salaries")
    plt.xlabel("Experiences (Months)")
    plt.ylabel("Salaries (Thousands)")
    plt.show()

def predict(w, b, predict_inp): # Takes the prediction input and predicts the output salary.
    predict = w * predict_inp + b

    return predict

main() # Call the main function to execute the code.