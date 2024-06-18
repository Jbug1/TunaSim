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

    if lower<=upper:
    
        return np.clip(element,lower,upper)
    
    #get back onside if crossed over
    else:
        return np.clip(element,lower,lower)
    
def tuna_combo_distance_demo(query,
                    target,
                    a = 0,
                    b = 1,
                    c = 0,
                    d = -np.inf,
                    e = np.inf,
                    f = 0,
                    g = 1,
                    h = 0,
                    i = -np.inf,
                    j = np.inf,
                    k = 0,
                    l = 1,
                    m = 0,
                    n = -np.inf,
                    o = np.inf,
                    p = 0,
                    q = 1,
                    r = 0,
                    s = 0,
                    t = -1,
                    u = 0,
                    v = -np.inf,
                    w = np.inf,
                    x = 0,
                    y = 1,
                    z = 0,
                    a_ = 0,
                    b_ = -1,
                    c_ = 0,
                    d_ = -np.inf,
                    e_ = np.inf,
                    f_ = 0,
                    g_ = 1,
                    h_ = 0,
                    i_ = 0,
                    j_ = -1,
                    k_ = 0,
                    l_ = -np.inf,
                    m_ = np.inf,
                    n_ = 0,
                    o_ = 1,
                    p_ = 0,
                    q_ = -np.inf,
                    r_ = np.inf,
                    s_ = -1,
                    t_ = 0,
                    u_ = 1,
                    v_ = 0,
                    w_ = -np.inf,
                    x_ = np.inf,
                    y_ = -1
                    ):
    """
    function of individual disagreements, sum_disagreement and length 
    constant and exponential for each
    knock-in for each
    knock-out for each
    Betas for each
    individual interactions for each
    clipping of interaction vector
    """

    #array to hold terms
    terms = np.zeros(6)

    ind_dif = None
    ind_add = None
    ind_mult = None

    if a != 0:
        dif = np.abs(query-target)
        ind_dif = (dif + 1e-50) ** b
        terms[1] += middle(a * np.sum(ind_dif) + c, d, e)

    #add always >0
    if f != 0:
        add = query+target
        ind_add = add ** g
        terms[1] += middle(f * np.sum(ind_add) + h, i, j)
        
    if k != 0:
        mult = query * target
        ind_mult = (mult + 1e-50) ** l
        terms[2] += middle(k * np.sum(ind_mult) + m, n, o)
    
    if p != 0:
        if ind_dif is None:
            dif = np.abs(query-target)
            ind_dif = (dif+1e-50) ** b
        if ind_add is None:
            add = query+target
            ind_add = add ** g

        if t >= 0:
            terms[3] += np.sum(middle((p * ind_dif ** q + r) * (s * ind_add ** t + u), v, w))
        else:
            terms[3] += np.sum(middle((p * ind_dif ** q + r) / (s * ind_add ** -t + u), v, w))

    if x != 0:
        if ind_dif is None:
            dif = np.abs(query-target)
            ind_dif = (dif + 1e-50) ** b

        if ind_mult is None:
            mult = query * target
            ind_mult = (mult+1e-50)**l

        if b_ >=0:
            terms[4] += np.sum(middle((x * ind_dif ** y + z) * (a_ *  ind_mult ** b_ + c_), d_, e_))
        else:
            terms[4] += np.sum(middle((x * ind_dif ** y + z) / (a_ *  ind_mult ** -b_ + c_), d_, e_))

    if f_ != 0:
        if ind_mult is None:
            mult = query * target
            ind_mult = (mult + 1e-50) ** l

        if ind_add is None:
            add = query+target
            ind_add = add ** g

        if j_ >=0:
            terms[5] += np.sum(middle((f_ * ind_mult ** g_ +h_) * (i_ * ind_add ** j_ + k_), l_, m_))
        else:
            terms[5] += np.sum(middle((f_ * ind_mult ** g_ +h_) * (i_ * ind_add ** -j_ + k_), l_, m_))

    #optional normalization by query and target
    if n_ != 0:

        if s_ >=0:
            terms *= np.sum(middle(n_ * query ** o_ + p_, q_, r_) + middle(n_ * query ** o_ + p_, q_, r_)) ** s_
        else:
            terms /= np.sum(middle(n_ * query ** o_ + p_, q_, r_) + middle(n_ * query ** o_ + p_, q_, r_)) ** -s_

    if t_ != 0:

        if y_>=0:
            terms *= np.sum(middle(t_ * query ** u_ + v_, w_, x_) * middle(t_ * query ** u_ + v_, w_, x_)) ** y_
        else:
            terms /= np.sum(middle(t_ * query ** u_ + v_, w_, x_) * middle(t_ * query ** u_ + v_, w_, x_)) ** -y_

    return np.sum(terms)
  
    
def tuna_combo_distance(query,
                    target,
                    a = 0,
                    b = 1,
                    c = 0,
                    d = -np.inf,
                    e = np.inf,
                    f = 0,
                    g = 1,
                    h = 0,
                    i = -np.inf,
                    j = np.inf,
                    k = 0,
                    l = 1,
                    m = 0,
                    n = -np.inf,
                    o = np.inf,
                    p = 0,
                    q = 1,
                    r = 0,
                    s = 0,
                    t = -1,
                    u = 0,
                    v = -np.inf,
                    w = np.inf,
                    x = 0,
                    y = 1,
                    z = 0,
                    a_ = 0,
                    b_ = -1,
                    c_ = 0,
                    d_ = -np.inf,
                    e_ = np.inf,
                    f_ = 0,
                    g_ = 1,
                    h_ = 0,
                    i_ = 0,
                    j_ = -1,
                    k_ = 0,
                    l_ = -np.inf,
                    m_ = np.inf,
                    n_ = 0,
                    o_ = 1,
                    p_ = 0,
                    q_ = -np.inf,
                    r_ = np.inf,
                    s_ = -1,
                    t_ = 0,
                    u_ = 1,
                    v_ = 0,
                    w_ = -np.inf,
                    x_ = np.inf,
                    y_ = -1,
                    z_ = 1
                    ):
    """
    function of individual disagreements, sum_disagreement and length 
    constant and exponential for each
    knock-in for each
    knock-out for each
    Betas for each
    individual interactions for each
    clipping of interaction vector
    """

    #array to hold terms
    terms = np.zeros(6)

    ind_dif = None
    ind_add = None
    ind_mult = None

    if a != 0:
        dif = np.abs(query-target)
        ind_dif = (dif + 1e-50) ** b
        terms[1] += middle(a * np.sum(ind_dif) + c, d, e)

    #add always >0
    if f != 0:
        add = query+target
        ind_add = add ** g
        terms[1] += middle(f * np.sum(ind_add) + h, i, j)
        
    if k != 0:
        mult = query * target
        ind_mult = (mult + 1e-50) ** l
        terms[2] += middle(k * np.sum(ind_mult) + m, n, o)
    
    if p != 0:
        if ind_dif is None:
            dif = np.abs(query-target)
            ind_dif = (dif+1e-50) ** b
        if ind_add is None:
            add = query+target
            ind_add = add ** g

        if t >= 0:
            terms[3] += np.sum(middle((p * ind_dif ** q + r) * (s * ind_add ** t + u), v, w))
        else:
            terms[3] += np.sum(middle((p * ind_dif ** q + r) / (s * ind_add ** -t + u), v, w))

    if x != 0:
        if ind_dif is None:
            dif = np.abs(query-target)
            ind_dif = (dif + 1e-50) ** b

        if ind_mult is None:
            mult = query * target
            ind_mult = (mult+1e-50)**l

        if b_ >=0:
            terms[4] += np.sum(middle((x * ind_dif ** y + z) * (a_ *  ind_mult ** b_ + c_), d_, e_))
        else:
            terms[4] += np.sum(middle((x * ind_dif ** y + z) / (a_ *  ind_mult ** -b_ + c_), d_, e_))

    if f_ != 0:
        if ind_mult is None:
            mult = query * target
            ind_mult = (mult + 1e-50) ** l

        if ind_add is None:
            add = query+target
            ind_add = add ** g

        if j_ >=0:
            terms[5] += np.sum(middle((f_ * ind_mult ** g_ +h_) * (i_ * ind_add ** j_ + k_), l_, m_))
        else:
            terms[5] += np.sum(middle((f_ * ind_mult ** g_ +h_) * (i_ * ind_add ** -j_ + k_), l_, m_))

    #optional normalization by query and target
    if n_ != 0:

        if s_ >=0:
            terms *= np.sum(middle(n_ * query ** o_ + p_, q_, r_) + middle(n_ * query ** o_ + p_, q_, r_)) ** s_
        else:
            terms /= np.sum(middle(n_ * query ** o_ + p_, q_, r_) + middle(n_ * query ** o_ + p_, q_, r_)) ** -s_

    if t_ != 0:

        if y_>=0:
            terms *= np.sum(middle(t_ * query ** u_ + v_, w_, x_) * middle(t_ * query ** u_ + v_, w_, x_)) ** y_
        else:
            terms /= np.sum(middle(t_ * query ** u_ + v_, w_, x_) * middle(t_ * query ** u_ + v_, w_, x_)) ** -y_

    return sigmoid(z_*np.sum(terms))
    
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
        total_disagreement = np.sum(dif)+1e-30
        terms[0] += middle(a*(total_disagreement+b)**c,d,e)
    
    if f != 0:
        ind_disagreements = np.sum((dif+1e-30)**h)
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

    return sigmoid(np.sum(terms))

    
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
