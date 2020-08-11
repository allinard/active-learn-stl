from STL import STLFormula
import operator as operatorclass
import pulp as plp
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import random


#CONSTANTS
M = 100000
M_up = 100000
M_low = 0.000001



def generate_signal_milp_quantitative(phi,start,rand_area,dimensions,U,epsilon,OPTIMIZE_ROBUSTNESS):
    """
        Function generating a signal satisfying an STL Formula.
        Takes as input:
            * phi: an STL Formula
            * start: a vector of the form [x0,y0,...] for the starting point coordinates
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the STLFormula is defined, e.g. dimensions=['x','y'].
            * U: a basic control policy standing for how much can move in 1 time stamp, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
            * epsilon: basic control policy parameter
            * OPTIMIZE_ROBUSTNESS: a flag whether the robustness of the generated signal w.r.t. phi has to be maximized or not
        The encoding details of the MILP optimization problem follows the quantitative enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """    
    dict_vars = {}
  
    #objective, maximize robustness
    rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Continuous')
    dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)] = rvar
            
    #Initialize model
    if OPTIMIZE_ROBUSTNESS:
        opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
        opt_model += rvar
    else:
        opt_model = plp.LpProblem("MIP Model")
    

    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(len(dimensions))),rand_area[0],rand_area[1],plp.LpContinuous)

    #the start is specified
    for dim in range(len(dimensions)):
        opt_model += s[0][dim] == start[dim]
    
    #basic control policy, i.e. how much can move in 1 time stamp
    #\forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-epsilon,U+epsilon)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-epsilon,U+epsilon)
        
        
    #recursive function
    def model_phi(phi,t,opt_model):
        if isinstance(phi, STLFormula.Predicate):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu == rvar
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu == rvar
            
        elif isinstance(phi, STLFormula.TrueF):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            opt_model += rvar >= M            
            
        elif isinstance(phi, STLFormula.FalseF):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            opt_model += rvar <= -M
            
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)
            
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)

        elif isinstance(phi,STLFormula.Always):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                     
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Eventually):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar >= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Negation):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            model_phi(phi.formula,t,opt_model)
            try:
                rvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                rvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)] = rvar_i
            opt_model += rvar == -rvar_i
    
    
    model_phi(phi,phi.horizon,opt_model)
    rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)]
    opt_model += rvar >= 0 
    
    opt_model.solve(plp.GUROBI_CMD(msg=False))

    if s[0][0].varValue == None:
        raise Exception("")
    
    return [[s[j][i].varValue for i in range(len(dimensions))] for j in range(phi.horizon+1)]
    
    
    


def generate_signal_milp_boolean(phi,start,rand_area,dimensions,U,epsilon):
    """
        Function generating a signal satisfying an STL Formula.
        Takes as input:
            * phi: an STL Formula
            * start: a vector of the form [x0,y0,...] for the starting point coordinates
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the STLFormula is defined, e.g. dimensions=['x','y'].
            * U: a basic control policy standing for how much can move in 1 time stamp, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
            * epsilon: basic control policy parameter
        The encoding details of the MILP optimization problem follows the boolean enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """    
    dict_vars = {}
  
    #satisfaction of phi
    zvar1 = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi))+'_t_'+str(phi.horizon)] = zvar1

    opt_model = plp.LpProblem("MIP Model")
  
            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(len(dimensions))),rand_area[0],rand_area[1],plp.LpContinuous)
       
    #the start is specified
    for dim in range(len(dimensions)):
        opt_model += s[0][dim] == start[dim]
    
    #control policy
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-epsilon,U+epsilon)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-epsilon,U+epsilon)   
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.FalseF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 0
        if isinstance(phi, STLFormula.Predicate):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) <= M_up*(1-zvar)-M_low
        elif isinstance(phi, STLFormula.Negation):
            model_phi1(phi.formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)] = zvar1 
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar <= zvar1
            opt_model += zvar <= zvar2
            opt_model += zvar >= 1-2+zvar1+zvar2
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar >= zvar1
            opt_model += zvar >= zvar2
            opt_model += zvar <= zvar1+zvar2
        elif isinstance(phi,STLFormula.Always):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                opt_model += zvar <= zvar_i
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
        elif isinstance(phi,STLFormula.Eventually):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    model_phi1(phi,phi.horizon,opt_model)
    
    opt_model += zvar1 == 1
        
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    return [[s[j][i].varValue for i in range(len(dimensions))] for j in range(phi.horizon+1)]

    
    
    
    
if __name__ == '__main__':

    #CONSTANTS
    INDEX_X = 0
    INDEX_Y = 1
    dimensions = ['x','y']

    #Definition of STL Formulae
    predicate_x_gt0 = STLFormula.Predicate('x',operatorclass.gt,0,INDEX_X)
    predicate_x_le1 = STLFormula.Predicate('x',operatorclass.le,1,INDEX_X)
    predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
    predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    eventually1 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt0,predicate_x_le1) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),10,20)
    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt0 = STLFormula.Predicate('y',operatorclass.gt,0,INDEX_Y)
    predicate_y_le1 = STLFormula.Predicate('y',operatorclass.le,1,INDEX_Y)
    eventually2 = STLFormula.Eventually(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt0,predicate_y_le1) ),30,50)
    predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
    predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
    predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
    predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
    always = STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,50) 
    phi = STLFormula.Conjunction(STLFormula.Conjunction(eventually1,eventually2),always)
    phi_nnf = STLFormula.toNegationNormalForm(phi,False)

    #parameters
    start=[0, 0]
    rand_area=[0, 4]
    U = 0.2
    epsilon = 0.05
    
    #generation of 3 trajectories (quantitative no maximization, quantitative with maximization, boolean)
    trajectory1 = generate_signal_milp_quantitative(phi_nnf,start,rand_area,dimensions,U,epsilon,False)
    trajectory2 = generate_signal_milp_quantitative(phi_nnf,start,rand_area,dimensions,U,epsilon,True)
    trajectory3 = generate_signal_milp_boolean(phi_nnf,start,rand_area,dimensions,U,epsilon)

    #Plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
    verts4_1 = [
        (0., 3.), # left, bottom
        (0., 4.), # left, top
        (1., 4.), # right, top
        (1., 3.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_1 = Path(verts4_1, codes)
    patch4_1 = patches.PathPatch(path4_1, facecolor='honeydew',lw=0)
    verts4_2 = [
        (2., 1.), # left, bottom
        (2., 2.), # left, top
        (3., 2.), # right, top
        (3., 1.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_2 = Path(verts4_2, codes)
    patch4_2 = patches.PathPatch(path4_2, facecolor='mistyrose',lw=0)
    verts4_3 = [
        (3., 0.), # left, bottom
        (3., 1.), # left, top
        (4., 1.), # right, top
        (4., 0.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_3 = Path(verts4_3, codes)
    patch4_3 = patches.PathPatch(path4_3, facecolor='honeydew',lw=0)
    ax.add_patch(patch4_1)
    ax.add_patch(patch4_2)
    ax.add_patch(patch4_3)
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plt.axis([rand_area[0]-0.2, rand_area[1]+0.2, rand_area[0]-0.2, rand_area[1]+0.2])
    plt.grid(True)

    ax.plot([x for (x, y) in trajectory1], [y for (x, y) in trajectory1], '-g', marker='o', label=r'quantitave $\rho='+str(round(phi.robustness(trajectory1,0),3))+'$')
    plt.grid(True)

    ax.plot([x for (x, y) in trajectory2],[y for (x, y) in trajectory2], '-b', marker='o', label=r'quantitave optimized $\rho='+str(round(phi.robustness(trajectory2,0),3))+'$')
    plt.grid(True)     
    
    ax.plot([x for (x, y) in trajectory3],[y for (x, y) in trajectory3], '-r', marker='o', label=r'boolean $\rho='+str(round(phi.robustness(trajectory3,0),3))+'$')
    plt.grid(True)                    

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    plt.show()

