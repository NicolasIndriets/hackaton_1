#!/usr/bin/env python
# coding: utf-8

# # Hackathon 1: simulations, descriptive stats & estimation
# 
# This project illustrates the course LEPL1109 with an industrial applications of simulations. You will analyse the performance of an assembly chain, analytically and with simulations. This is an introduction to supply chain management which is a field actively studied by engineers (see e.g. LINMA2470 Stochastic modelling). 
# 
# ## Report content
# 
# •	Grades are granted to the members whose names are in the Jupyter notebook. If your name doesn’t appear on the top of the notebook, you’ll get a 0, even though you are in a group on Moodle.
# 
# •	The jupyter notebook must be compiled with printed results and next submitted via moodle. The absence of compiled results (or non-printed values) leads to a lower grade.
# 
# ## Report submission
# 
# •	Deadline, see moodle website. Submission after the deadline will not be accepted.
# 
# •	To submit your report, go to the section “APP” on Moodle and the subsection “Soumission du rapport”. You can upload your work there. Once you are sure that it is your final version, click the button “Envoyer le devoir”. It is important that you don’t forget to click on this button ! 
# 
# •	Reports that have not been uploaded through Moodle will not be corrected.
# 
# 
# ## Names and Noma of participants:
# 
# 
# Part. 1: Indriets Nicolas 58541800 
# 
# Part. 2: De Jaegher Nicolas 56691800
# 
# Part. 3: Hallard Zoe 54341900
# 
# Part. 4: Macq Sebastien 66061900
# 
# Part. 5: Tydgat Brice 
# 
# Part. 6: Wlodarski Lisa 05971900

# In a manufacture of mechanical devices for the automotive industry, the production of a gearbox is done by a line of 3 successive working stations. The production of a new gearbox starts only when the previous one is fully assembled. In what follows, the assembly time of one gearbox is denoted by $T$. Also, $T_k$ is the assembly time at machine $k$. Finally $\mu_k = \mathbb{E}(T_k)$ and $\sigma^2_k = \mathbb{V}(T)$. Details about the production line are provided in the graph below.
# 
# ![Capture.PNG](attachment:Capture.PNG)
# 
# where $p=0.75$ indicates for example that the output of machine 1 ends up in machine 2 with probability $0.75$. 
# 
# ------------------------------------------------------------------------------------------------------
# 
# 1.	For each machine $k$, choose a (non-Gaussian) statistical distribution for the assembly time $T_k$ and provide value(s) for its parameter(s).
# 
# ------------------------------------------------------------------------------------------------------

# machine 1: gamma (alpha = 49/4, betha = 4/7)
# machine 2: exponential (betha = 12)
# machine 3: gamma (alpha = 4, betha = 4)
# machine 4: exponential (betha = 8)
# 
# 

# ------------------------------------------------------------------------------------------------------------------------
# 
# b) Calculate the expected time for assembling one gearbox, $\mathbb{E}(T)$.
# 
# ------------------------------------------------------------------------------------------------------------------------

# E = 7 + 0.75 * 12  + 0.25 * 16 + 8 =28
# 

# ------------------------------------------------------------------------------------------------------
# 
# 2. Simulate 5000 durations  $T$  for assembling one single gearbox and report the mean (check that its is well close to the expectation of question 1), standard deviation, 5% and 95% percentile. Plot an histogram. Save the simulated durations in a file "sample.csv".
# 
# ------------------------------------------------------------------------------------------------------

# In[31]:


import numpy as np
shape1, scale1 = 49/4, 4/7  # mean=4, std=2*sqrt(2)
machine1 = np.random.gamma(shape1, scale1, 5000)
#alpha = k = shape , betha = theta = scale, 

scale2 = 12
machine2 = np.random.exponential(scale2, 3750)

shape3, scale3 = 4, 4
machine3 = np.random.gamma(shape3, scale3, 1250)

scale4 = 8
machine4 = np.random.exponential(scale4, 5000)

machine23 = np.append(machine2, machine3)

globalmachine = machine1 + machine23 + machine4

globaltime = np.sum(globalmachine)
meantime = globaltime/5000
print(meantime)

import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50, density=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()


# 
# ------------------------------------------------------------------------------------------------------
# 
# 
# 3. a) Load simulated data from the file "sample.csv". Fit the exponential, gamma and chi-square distributions and estimate their parameters by log-likelihood maximization. Select the best non-Gaussian statistical distribution for  $T$. Justify your choice.
# 
# Rem: if you use scipy, set the location parameter to zero with option 'floc=0'.
# 
# ------------------------------------------------------------------------------------------------------

# Answer here
# 

# In[2]:


#code here


# b) During the lectures, we have seen that the chi-square distribution has only one single parameter (the number of degrees of freedom). In Scipy, we find two other parameters location "loc" and scale "scale". Why? 

# Answer here
# 

# ------------------------------------------------------------------------------------------------------
# 
# c) Compare the empirical distribution with the distribution you chose at 3a). For this purpose, plot an histogram of  durations and add the probability density function (pdf) that you have selected for modelling the random variable.
# 
# ------------------------------------------------------------------------------------------------------

# In[3]:


#code here


# ------------------------------------------------------------------------------------------------------
# 
# 4. We want to fit a Weibull distribution to $T$. The density, $f(t)$, depends upon two parameters $\alpha>0$ and $\lambda>0$ and is given by:
# 
# $$ f(t)=\lambda \alpha t^{\alpha-1} \exp(-\lambda t^\alpha) $$
# 
# a) Write two functions: one for the density and one that computes the log-likelihood of a sample of durations for this distribution. Test this function on the simulated dataset and for $\alpha=1$ and $\lambda=\frac{1}{28}$
# 
# ------------------------------------------------------------------------------------------------------

# In[7]:


#code here
import math
import numpy as np
def density(alpha, lamda, t):
    l = lamda
    a = alpha
    f = l*a*t**(a-1)*math.exp(-l*t**a)
    return f

def log_likelihood(data, alpha, lamda):
    M = []
    for x in data:
        M.append(np.log(density(alpha, lamda, x)))
    return np.sum(M)  
    #faire somme des ln des f(t) ==> f(t) est un gamma?
print(log_likelihood([1,2,3,4],1,2))


# # ------------------------------------------------------------------------------------------------------
# 
# b) Maximize the log-likelihood to find estimates $\hat{\alpha}$ and $\hat{\lambda}$. Next compare the log-likelihood with those computed previously. Is this model better?
# 
# Hint: use the "scipy.optimize" package and the function "minimize". To avoid adding constraints, you can do a change of variables: $\alpha=\exp(a)$ and $\lambda=\exp(b)$. The function "minimize" may send back a warning (division by zero in log) but converges well to the solution.
# 
# ------------------------------------------------------------------------------------------------------

# In[5]:


#code here


# ------------------------------------------------------------------------------------------------------
# 
# c) The expectation and variance of the Weibull distribution are 
# $$ \mathbb{E}(T)=\lambda^{-\frac{1}{\alpha}}\Gamma(\frac{1}{\alpha}+1) $$
# 
# $$ \mathbb{V}(T)=\lambda^{-\frac{2}{\alpha}} \left(  \Gamma(\frac{2}{\alpha}+1) -\Gamma(\frac{1}{\alpha}+1)^2  \right) $$
# 
# Estimate $\alpha$  and $\lambda$ by the methods of moments. Next calculate the log-likelihood with these estimates and compare with these obtained by log-likelihood maximization. Is the result surprising ?
# 
# Hint: use the scipy.optimize package and the function "root". The gamma function is scipy.special.gamma(z)
# 
# ------------------------------------------------------------------------------------------------------

# In[6]:


#code here


# ------------------------------------------------------------------------------------------------------
# 
# 5.	Based on the content of the “Self-learning 2”, Bootstrap 1000 times the sample of durations simulated in Question 2 (stored in the file "sample.csv") and compute, the mean, standard deviation and 95% and 90% confidence interval for parameter(s) estimate(s). Plot the empirical distribution of parameter(s).
# 
# ------------------------------------------------------------------------------------------------------

# In[7]:


#code here


# -----------------------------------------------------------------------------------------------------
# 
# 6. Using simulated assembling times, determine the probability that assembling 10 gearboxes lasts less than 320 seconds $P(T\leq 320\,sec)$. 
# 
# ------------------------------------------------------------------------------------------------------

# In[8]:


#code here


# ------------------------------------------------------------------------------------------------------
# 
# 7. Is there any alternative to numerical computing in question 6 for approximating  $P(T\leq 320\,sec)$? Please detail your answer and provide an estimate of this probability.
# 
# ------------------------------------------------------------------------------------------------------

# Answer here
# 

# In[ ]:




