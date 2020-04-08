from STL import STLFormula
import operator as operatorclass
import pulp as plp



#CONSTANTS
M_up = 100000
M_low = 0.000001

#HARDCODED
#TODO: manage more dimensions
NB_DIMENSIONS = 2


def directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area):
    """
        Function computing the directed pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
        The encoding details of the MILP optimization problem follow Raman et al., "Model  predictive  control  with  signaltemporal logic specifications,” in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """

    dict_vars = {}
  
    #objective, maximize epsilon
    epsilon = plp.LpVariable(name='epsilon',cat='Continuous',lowBound=0,upBound=rand_area[1]-rand_area[0])
    
    #satisfaction of phi1 and phi2
    zvar1 = plp.LpVariable('z1_'+str(id(phi1))+'_t_'+str(phi1.horizon),cat='Binary')
    zvar2 = plp.LpVariable('z2_'+str(id(phi2))+'_t_'+str(phi2.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi1))+'_t_'+str(phi1.horizon)] = zvar1
    dict_vars['z2_'+str(id(phi2))+'_t_'+str(phi2.horizon)] = zvar2

    opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
    opt_model += epsilon
  
    dict_vars['epsilon'] = epsilon

            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(max(phi1.horizon,phi2.horizon)+1),range(NB_DIMENSIONS)),rand_area[0],rand_area[1],plp.LpContinuous)    
       
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
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
        elif isinstance(phi, STLFormula.Predicate):
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
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
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
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    def model_phi2(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.Negation):
            model_phi2(phi.formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t)] = zvar1 
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
        elif isinstance(phi, STLFormula.Predicate):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            if phi.operator == operatorclass.gt or phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu + epsilon <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) - epsilon <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu + epsilon <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) - epsilon <= M_up*(1-zvar)-M_low
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi2(phi.first_formula,t,opt_model)
            model_phi2(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z2_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar <= zvar1
            opt_model += zvar <= zvar2
            opt_model += zvar >= 1-2+zvar1+zvar2
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi2(phi.first_formula,t,opt_model)
            model_phi2(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z2_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z2_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar >= zvar1
            opt_model += zvar >= zvar2
            opt_model += zvar <= zvar1+zvar2
        elif isinstance(phi,STLFormula.Always):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi2(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar <= zvar_i
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
        elif isinstance(phi,STLFormula.Eventually):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi2(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z2_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] = zvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z2_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    model_phi1(phi1,phi1.horizon,opt_model)
    model_phi2(phi2,phi2.horizon,opt_model)

    zvar1 = dict_vars['z1_'+str(id(phi1))+'_t_'+str(phi1.horizon)]
    zvar2 = dict_vars['z2_'+str(id(phi2))+'_t_'+str(phi2.horizon)]
    
    opt_model += zvar1 == 1
    opt_model += zvar2 == 0
        
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    if epsilon.varValue == None:
        return 0.0
    return epsilon.varValue





def pompeiu_hausdorff_distance(phi1,phi2,rand_area):
    """
        Function computing the pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
    """
    return round(max(directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area),directed_pompeiu_hausdorff_distance(phi2,phi1,rand_area)),5)










if __name__ == '__main__':
    
    #Constant
    INDEX_X = 0
    INDEX_Y = 1
    
    #Examples of Madsen et al. (2018)
    predicate_x_ge02 = STLFormula.Predicate('x',operatorclass.ge,0.2,INDEX_X)
    predicate_x_lt04 = STLFormula.Predicate('x',operatorclass.le,0.4,INDEX_X)
    predicate_x_lt044 = STLFormula.Predicate('x',operatorclass.le,0.44,INDEX_X)

    phi1 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi2 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20)
    phi3 = STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
    phi4 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20),STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20))
    phi5 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,10),STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),12,20))
    phi6 = STLFormula.Always(STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,4),0,16)
    true = STLFormula.TrueF()


    #Reproduces the results of Madsen et al. (2018)
    rand_area = [0,1]
    
    print('d(True,phi1)',pompeiu_hausdorff_distance(true,phi1,rand_area),'(0.6)')
    print('d(phi1,phi1)',pompeiu_hausdorff_distance(phi1,phi1,rand_area),'(0.0)')
    print('d(phi1,phi2)',pompeiu_hausdorff_distance(phi1,phi2,rand_area),'(0.04)')
    print('d(phi1,phi3)',pompeiu_hausdorff_distance(phi1,phi3,rand_area),'(0.6)')
    print('d(phi1,phi4)',pompeiu_hausdorff_distance(phi1,phi4,rand_area),'(0.0)')
    print('d(phi1,phi5)',pompeiu_hausdorff_distance(phi1,phi5,rand_area),'(0.6)')
    print('d(phi1,phi6)',pompeiu_hausdorff_distance(phi1,phi6,rand_area),'(0.6)')

    print('')

    print('d(True,phi2)',pompeiu_hausdorff_distance(true,phi2,rand_area),'(0.56)')
    print('d(phi2,phi2)',pompeiu_hausdorff_distance(phi2,phi2,rand_area),'(0.0)')
    print('d(phi2,phi3)',pompeiu_hausdorff_distance(phi2,phi3,rand_area),'(0.56)')
    print('d(phi2,phi4)',pompeiu_hausdorff_distance(phi2,phi4,rand_area),'(0.04)')
    print('d(phi2,phi5)',pompeiu_hausdorff_distance(phi2,phi5,rand_area),'(0.56)')
    print('d(phi2,phi6)',pompeiu_hausdorff_distance(phi2,phi6,rand_area),'(0.56)')

    print('')

    print('d(True,phi3)',pompeiu_hausdorff_distance(true,phi3,rand_area),'(0.6)')
    print('d(phi3,phi3)',pompeiu_hausdorff_distance(phi3,phi3,rand_area),'(0.0)')
    print('d(phi3,phi4)',pompeiu_hausdorff_distance(phi3,phi4,rand_area),'(0.6)')
    print('d(phi3,phi5)',pompeiu_hausdorff_distance(phi3,phi5,rand_area),'(0.6)')
    print('d(phi3,phi6)',pompeiu_hausdorff_distance(phi3,phi6,rand_area),'(0.6)')

    print('')

    print('d(True,phi4)',pompeiu_hausdorff_distance(true,phi4,rand_area),'(0.6)')
    print('d(phi4,phi4)',pompeiu_hausdorff_distance(phi4,phi4,rand_area),'(0.0)')
    print('d(phi4,phi5)',pompeiu_hausdorff_distance(phi4,phi5,rand_area),'(0.6)')
    print('d(phi4,phi6)',pompeiu_hausdorff_distance(phi4,phi6,rand_area),'(0.6)')

    print('')

    print('d(True,phi5)',pompeiu_hausdorff_distance(true,phi5,rand_area),'(0.6)')
    print('d(phi5,phi5)',pompeiu_hausdorff_distance(phi5,phi5,rand_area),'(0.0)')
    print('d(phi5,phi6)',pompeiu_hausdorff_distance(phi5,phi6,rand_area),'(0.6)')

    print('')

    print('d(True,phi6)',pompeiu_hausdorff_distance(true,phi6,rand_area),'(0.6)')
    print('d(phi6,phi6)',pompeiu_hausdorff_distance(phi6,phi6,rand_area),'(0.0)')
    
    
    print('')
    print('')
    
    #Example motion planning
    
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
    phi_target = STLFormula.toNegationNormalForm(phi,False)

    phi_hypothesis = STLFormula.Conjunction(eventually1,eventually2)

    rand_area = [0,4]

    print(pompeiu_hausdorff_distance(true,phi_target,rand_area))
    print(pompeiu_hausdorff_distance(phi_hypothesis,phi_target,rand_area))
    print(pompeiu_hausdorff_distance(eventually1,eventually2,rand_area))


    
    