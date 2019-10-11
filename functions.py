import numpy as np

def logistic_loss(x, y, beta):
    return np.log(1 + np.exp(x.dot(beta))) - y*x.dot(beta)


def logistic_loss_multitask(x, y, beta):
    return np.sum([logistic_loss(x, y[i], beta[:, i]) for i in range(len(y))])

def logistic_loss_prime(x, y, beta):
    return x * ( np.exp(x.dot(beta)) / (1 + np.exp(x.dot(beta))) - y)

def logistic_loss_prime_multitask(x, y, beta):
    return np.array([
        logistic_loss_prime(x, y[i], beta[:, i]) for i in range(len(y))]).T

def linear_loss(x, y, beta):
    return 0.5*(y-x.dot(beta))**2

def linear_loss_prime(x, y, beta):
    return (-x)*(y-x.dot(beta))

def linear_loss_multitask(x, y, beta):
    return np.sum([
        linear_loss(x, y[i], beta[:, i]) for i in range(len(y))])

def linear_loss_prime_multitask(x, y, beta):
    return np.array([
        linear_loss_prime(x, y[i], beta[:, i]) for i in range(len(y))]).T

def lasso_penalty(beta, target):
    return np.linalg.norm(beta-target, ord=1)

def lasso_derivative(beta, target):
    return np.sign(beta - target)

def l2_penalty(x, target):
    return 0.5*np.linalg.norm(x - target, ord=2)

def l2_prime(x, target):
    return x - target




bad_vals = ["None", None, "not reported"]
def either_bad(x, y):
    if x in bad_vals or y in bad_vals:
        #print("Bad value: {},{}".format(x,y))
        return True
    else:
        #print("Values Fine")
        return False

def abs_diff(x, y):
    return np.abs(float(x) - float(y))

def discrete_diff(x, y):
    return float(x != y)


def safe_wrapper(x, y, f):
    if either_bad(x, y):
        return 0.
    else:
        #print("Trying {}".format(f))
        return f(x,y)
