import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy import stats
from scipy.optimize import minimize as mini
from scipy.optimize import approx_fprime as approx

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def middle(element,lower,upper):
    """ 
    return middle element
    """

    if element>=lower:
        if element<= upper:
            return element
        else:
            return upper
    else:
        return lower
    
def tuna_dif_distance(query,
                    target,
                    a = 0,
                    b = 0,
                    c = 1,
                    d = -np.inf,
                    e = np.inf,
                    f = 0,
                    g = 0,
                    h = 1,
                    i = -np.inf,
                    j = np.inf,
                    k = 0,
                    l = 0,
                    m = 1,
                    n = -np.inf,
                    o = np.inf,
                    p = 0,
                    q = 0,
                    r = 1,
                    s = -np.inf,
                    t = np.inf,
                    u = 0,
                    v = 0,
                    w = 1,
                    x = -np.inf,
                    y = np.inf,
                    z = 0,
                    a_ = 0,
                    b_ = 1,
                    c_ = -np.inf,
                    d_ = np.inf,
                    ):
    """
    function of individual disagreements, sum_disagreement and length 
    constant and exponential for each
    knock-in for each
    knock-out for each
    Betas for each
    interactions for each
    """
    dif = np.abs(query-target)

    #array to hold terms
    terms = np.zeros(6)
    
    #check who needs to be evaluated
    if a != 0:
        total_disagreement = np.sum(dif)+1e-20
        terms[0] += middle(a*(total_disagreement+b)**c,d,e)
    
    if f != 0:
        ind_disagreements = np.sum((dif+1e-20)**h)
        terms[1] += middle(f*(ind_disagreements+g),i,j)

    if k != 0:
        disagreement_length = len(dif)
        terms[2] += middle(k*(disagreement_length+l)**m,n,o)
    
    if p != 0:
        terms[3] += middle(p*(total_disagreement*ind_disagreements+q)**r,s,t) 

    if u != 0:
        terms[4] += middle(u*(total_disagreement*disagreement_length+v)**w,x,y)

    if z != 0:
        terms[5] += middle(z*(ind_disagreements*disagreement_length+a_)**b_,c_,d_)

    return np.sum(terms)

    
def tuna_dif_distance_old(query,
                    target,
                    a = 0,
                    b = 0,
                    c = 1,
                    d = -np.inf,
                    e = np.inf,
                    f = 0,
                    g = 0,
                    h = 1,
                    i = -np.inf,
                    j = np.inf,
                    k = 0,
                    l = 0,
                    m = 1,
                    n = -np.inf,
                    o = np.inf,
                    p = 0,
                    q = 0,
                    r = 1,
                    s = -np.inf,
                    t = np.inf,
                    u = 0,
                    v = 0,
                    w = 1,
                    x = -np.inf,
                    y = np.inf,
                    z = 0,
                    a_ = 0,
                    b_ = 1,
                    c_ = -np.inf,
                    d_ = np.inf,
                    ):
    """
    function of individual disagreements, sum_disagreement and length 
    constant and exponential for each
    knock-in for each
    knock-out for each
    Betas for each
    interactions for each
    """
    dif = np.abs(query-target)
    
    #check who needs to be evaluated
    total_disagreement = np.sum(dif)+1e-20
    ind_disagreements = np.sum((dif+1e-20)**h)
    disagreement_length = len(dif)

    individuals = middle(a*(total_disagreement+b)**c,d,e) + middle(f*(ind_disagreements+g),i,j) + middle(k*(disagreement_length+l)**m,n,o)
    interactions = middle(p*(total_disagreement*ind_disagreements+q)**r,s,t) \
                   + middle(u*(total_disagreement*disagreement_length+v)**w,x,y) \
                   + middle(z*(ind_disagreements*disagreement_length+a_)**b_,c_,d_)

    return sigmoid(individuals + interactions)


def tuna_dif_distance_double_old(query,
                    target,
                    a = 0,
                    b = 1,
                    c = -np.inf,
                    d = np.inf,
                    e = 0,
                    f = 1,
                    g = -np.inf,
                    h = np.inf,
                    i = 0,
                    j = 1,
                    k = -np.inf,
                    l = np.inf,
                    m = 0,
                    n = 0,
                    o = 0,
                    ):
    """
    function of individual disagreements, sum_disagreement and length 
    constant and exponential for each
    knock-in for each
    knock-out for each
    Betas for each
    interactions for each
    """
    dif = np.abs(query-target)

    #do these before so we can get interactions with division
    #may have to add jitter later
    
    total_disagreement = np.sum(dif)+1e-10
    ind_disagreements = np.sum((dif+1e-10)**f)
    disagreement_length = len(dif)

    individuals = middle(a*total_disagreement**b,c,d) + middle(e*ind_disagreements,g,h) + middle(i*disagreement_length**j,k,l)
    interactions = m*total_disagreement*ind_disagreements + n*total_disagreement*disagreement_length + o*ind_disagreements*disagreement_length

    return sigmoid(individuals + interactions)

def tuna_plus_distance(query,
                       target,
                        a = 0,
                        b = 0,
                        c = 0,
                        d = 0,
                        e = 0,
                        f = 0,
                        g = 0,
                        h = 0,
                        i = 0,
                        j = 0,
                        k = 0,
                        l = 0,
                        m = 0,
                        n = 0,
                        o = 0,
                        p = 0,
                        q = 0,
                        r = 0,
                        s = 0,
                        t = 0,
                        u = 0,
                        v = 0,
                        w = 0,
                        x = 0,
                        y = 0,
                        z = 0
                        ):

    merged = query + target
    concat = np.concatenate([query, target])
    merged_ent = a*stats.entropy(merged)**b
    norm_ent = c*stats.entropy(concat)**d
    query_ent = e*stats.entropy(query)**f
    target_ent = g*stats.entropy(target)**h
    len_merged = i*len(merged)**j
    len_concat = k*len(concat)**l


    ind_terms = (middle(merged_ent,m,n)
                 + middle(norm_ent,o,p)
                 + middle(query_ent,q,r)
                 + middle(target_ent,s,t)
                 + middle(len_merged,u,v)
                 + middle(len_concat,w,x)
    )
                
    interactions = y*merged_ent*norm_ent + z*merged_ent*len_merged 

    return ind_terms + interactions
