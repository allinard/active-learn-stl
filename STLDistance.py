from STL import STLFormula
import operator as operatorclass
import pulp as plp
from itertools import product, tee, accumulate, combinations, chain
import copy


#CONSTANTS
M_up = 100000
M_low = 0.000001


def directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area,dimensions):
    """
        Function computing the directed pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the 2 STLFormulae are defined, e.g. dimensions=['x','y'].
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
    s = plp.LpVariable.dicts("s",(range(max(phi1.horizon,phi2.horizon)+1),range(len(dimensions))),rand_area[0],rand_area[1],plp.LpContinuous)    
       
       
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
        elif isinstance(phi, STLFormula.FalseF):
            try:
                zvar = dict_vars['z2_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z2_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z2_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 0
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





def pompeiu_hausdorff_distance(phi1,phi2,rand_area,dimensions):
    """
        Function computing the pompeiu-hausdorff distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the 2 STLFormulae are defined, e.g. dimensions=['x','y'].
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
    """
    return round(max(directed_pompeiu_hausdorff_distance(phi1,phi2,rand_area,dimensions),directed_pompeiu_hausdorff_distance(phi2,phi1,rand_area,dimensions)),5)




def get_cuboids(bcset):
    cuboids = []
    if isinstance(bcset.lst,N_Cuboid):
        return bcset
    if type(bcset.lst) == type(None):
        return cuboids
    for elt in bcset.lst:
        if isinstance(elt,N_Cuboid):
            cuboids.append(elt)
        else:
            cuboids.extend(get_cuboids(elt))
        
    return cuboids



def recursive_substract(cuboids,bcset):
    if isinstance(bcset,N_Cuboid):
        rest_of_a = [bcset]
        for b in cuboids:
            toadd = []
            todel = []
            for a_prime in rest_of_a:
                todel.append(a_prime)
                toadd.extend(a_prime - b)
            for d in todel:
                rest_of_a.remove(d)
            rest_of_a.extend(toadd)
        sub = list(set(rest_of_a))
        if len(sub)==1:
            return sub[0]
        if sub:
            return BoxSet(sub)
        return None
        
    elif isinstance(bcset,BoxSet):
        liste = []
        if type(bcset.lst) == type(None):
            return None
        for elt in bcset.lst:
            substract = recursive_substract(cuboids,elt)
            if type(substract) != type(None):
                liste.append(substract)

        if liste:
            return BoxSet(liste)
        return None
    
    elif isinstance(bcset,ChoiceSet):
        liste = []
        for elt in bcset.lst:
            substract = recursive_substract(cuboids,elt)
            if type(substract) != type(None):
                liste.append(substract)
        if liste:
            return ChoiceSet(liste)
        return None
    
    else:
        print("ERROR")
        exit()



def refactor_aos(bcset):
    if isinstance(bcset,N_Cuboid):
        return bcset

    lst = []
    try:
        for elt in bcset.lst:
            resrefact = refactor_aos(elt)
            if type(resrefact) is bool:
                continue
            else:
                lst.append(resrefact)
    except TypeError:
        pass
    except AttributeError:
        return False
    lst = list(set(lst))
    if not lst:
        return False
    if len(lst)==1:
        return lst[0]
    
    if isinstance(bcset,BoxSet):
        return BoxSet(lst)
    else:
        return ChoiceSet(lst)




#Definition of an iterator
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)



class N_Cuboid:
    
    def __init__(self,dict_dimensions,dimensions):
        self.dict_dimensions = dict_dimensions
        self.dimensions = dimensions
        
    def __str__(self):
        return str(self.dict_dimensions)
    
    def show(self,prefix=''):
        print(prefix,self.dict_dimensions)
    
    def __eq__(self, other):
        return self.dict_dimensions == other.dict_dimensions

    def __ne__(self, other):
        return self.dict_dimensions != other.dict_dimensions

    def __hash__(self):
        return hash(str(self.dict_dimensions))

    def __iter__(self):
        return iter([self])
    
    def __and__(self, other):
        dict_dimensions_intersection = {}
        for dimension in self.dimensions:
            x1 = max(self.dict_dimensions[dimension][0],other.dict_dimensions[dimension][0])
            x2 = min(self.dict_dimensions[dimension][1],other.dict_dimensions[dimension][1])
            dict_dimensions_intersection[dimension] = (x1,x2)
            if x1>x2:
                return None
        return type(self)(dict_dimensions_intersection,self.dimensions)


    def __sub__(self, other):
        
        intersection = self & other
                
        if intersection is None:
            return [self]
        else:
        
            dict_process = {}
            
            for dimension in self.dimensions:
                dict_process[dimension] = {self.dict_dimensions[dimension][0],self.dict_dimensions[dimension][1]}
                if self.dict_dimensions[dimension][0] < other.dict_dimensions[dimension][0] < self.dict_dimensions[dimension][1]:
                    dict_process[dimension].add(other.dict_dimensions[dimension][0])
                if self.dict_dimensions[dimension][0] < other.dict_dimensions[dimension][1] < self.dict_dimensions[dimension][1]:
                    dict_process[dimension].add(other.dict_dimensions[dimension][1])
            
            arguments = [pairwise(sorted(dict_process[dimension])) for dimension in self.dimensions]
            difference = []
            for truc in product(*arguments):
                iterator = iter(truc)
                dict_dimensions_diff = {}
                for dimension in self.dimensions:
                    dict_dimensions_diff[dimension] = next(iterator)
                instance = type(self)(dict_dimensions_diff,self.dimensions)
                if instance != intersection:
                    difference.append(instance)
            
            return difference


    def __or__(self, other):
        return list(set( list(self - other) + list(other - self) ))
        
        
    def __add__(self,other):
        if isinstance(other,N_Cuboid):
            intersection = self & other
            if intersection is None:
                return [self,other]
            else:
                return list(set( list(self - other) + [other] ))
        elif isinstance(other,BoxSet):
            union = {}
            if not other.lst:
                return [self]
            for a in other.lst:
                res = a+self
                if isinstance(res,BoxSet) or isinstance(res,ChoiceSet):
                    union[res] = None
                else:
                    for rect in res:
                        union[rect] = None
            return BoxSet(list(union))



    @property
    def n_dim_volume(self):
        volume = 1
        for dimension in self.dict_dimensions:
            volume = volume * (self.dict_dimensions[dimension][1] - self.dict_dimensions[dimension][0])
        return volume



class N_Cuboid_Predicate(N_Cuboid):
    pass


class N_Cuboid_Timed(N_Cuboid):
    pass


class N_Cuboid_TrueFalse(N_Cuboid):
    pass







class ChoiceSet:
    """
    Class representing a choice set
    """
    def __init__(self,lst):
        self.lst = lst
    
    def __str__(self):
        return str(self.lst)
    
    def show(self,prefix=''):
        print(prefix,"ChoiceSet")
        for i in self.lst:
            i.show(prefix+'\t')

    def __hash__(self):
        return hash(str(self))

    def __eq__(self,other):
        return self.lst == other.lst

    def __iter__(self):
        return iter(self.lst)

    def __add__(self,other):
        union = {} 
        if isinstance(other,ChoiceSet):
            if not other.lst:
                return [self]
            if not self.lst:
                return [other]
            for a,b in product(self.lst,other.lst):
                res = a+b
                if type(res).__name__ == "NoneType":
                    continue
                if isinstance(res,ChoiceSet) or isinstance(res,BoxSet):
                    union[res] = None
                else:
                    for rect in res:
                        union[rect] = None
            return ChoiceSet(list(union))
        elif isinstance(other,N_Cuboid):
            if not self.lst:
                return [other]
            for a in self.lst:
                if isinstance(a,BoxSet) or isinstance(a,ChoiceSet):
                    if not a.lst:
                        continue
                res = a+other
                if isinstance(res,BoxSet) or isinstance(res,ChoiceSet):
                    union[res] = None
                else:
                    for rect in res:
                        union[rect] = None
            return ChoiceSet(list(union))
        else:
            if not other.lst:
                return [self]
            if not self.lst:
                return [other]
            cuboids = get_cuboids(other)
            new_f1 = recursive_substract(cuboids,self)
            if type(new_f1) != type(None):
                return BoxSet([other,new_f1])
            return [other]    
            
    def __sub__(self,other):
        sub = []
        if not self.lst:
            return []
            
        if isinstance(other,N_Cuboid):
            rest_of_a = []
            for a in self.lst:
                rest_of_a.append(a)
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - other)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
            sub.extend(rest_of_a)
            return ChoiceSet(list(set(sub)))
            
        if not other.lst:
            return [self]
        if self.lst == other.lst:
            return ChoiceSet([])
                
        rest_of_a = []
        for a in self.lst:
            rest_of_a.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
        
        sub.extend(rest_of_a)
        return ChoiceSet(list(set(sub)))
            
    def __or__(self,other):
        sd = []
        if not self.lst:
            return other
        if not other.lst:
            return self
        if self.lst == other.lst:
            return ChoiceSet([])
                
        rest_of_a = []
        for a in self.lst:
            rest_of_a.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
        
        sd.extend(rest_of_a)   
        
        
        rest_of_b = []
        for b in other.lst:
            rest_of_b.append(b)
            for a in self.lst:
                toadd = []
                todel = []
                for b_prime in rest_of_b:
                    todel.append(b_prime)
                    toadd.extend(b_prime - a)
                for d in todel:
                    rest_of_b.remove(d)
                rest_of_b.extend(toadd)
        
        sd.extend(rest_of_b)    
        
        return ChoiceSet(list(set(sd)))


    def __floordiv__(self,other):
        div = []
        if not self.lst:
            return other.lst
        if not other.lst:
            return self.lst
        if self.lst == other.lst:
            return []


        rest_of_a = BoxSet([])
        for a in self.lst:
            rest_of_a.lst.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a.lst:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.lst.remove(d)
                rest_of_a = rest_of_a+BoxSet(toadd)
        
        div.extend(rest_of_a.lst)   

        rest_of_b = BoxSet([])
        for b in other.lst:
            rest_of_b.lst.append(b)
            for a in self.lst:
                toadd = []
                todel = []
                for b_prime in rest_of_b.lst:
                    todel.append(b_prime)
                    toadd.extend(b_prime - a)
                for d in todel:
                    rest_of_b.lst.remove(d)
                rest_of_b = rest_of_b+BoxSet(toadd)
        
        div.extend(rest_of_b.lst)    

        return div








class BoxSet:
    """
    Class representing a box set
    """
        
    def __init__(self,lst):
        self.lst = lst
    
    def __str__(self):
        return str(self.lst)
    
    def show(self,prefix=''):
        print(prefix,"BoxSet")
        for i in self.lst:
            i.show(prefix+'\t')

    def __hash__(self):
        return hash(str(self))

    def __eq__(self,other):
        return self.lst == other.lst

    def __iter__(self):
        return iter(self.lst)
        
    def __add__(self,other):
    
        union = {}
        if isinstance(other,BoxSet):
            if not self.lst:
                return other
            if not other.lst:
                return self
            try:
                for a,b in product(self.lst,other.lst):
                    res = a+b
                    if type(res).__name__=="NoneType":
                        continue
                    if isinstance(res,BoxSet) or isinstance(res,ChoiceSet):
                        union[res] = None
                    else:
                        for rect in res:
                            union[rect] = None
            except TypeError:
                for a,b in product([self.lst],other.lst):
                    for rect in a+b:
                        union[rect] = None
            return BoxSet(list(union))
        elif isinstance(other,N_Cuboid):
            if not self.lst:
                return other
            try:
                for a in self.lst:
                    res = a+other
                    if isinstance(res,BoxSet) or isinstance(res,ChoiceSet):
                        union[res] = None
                    elif isinstance(res,N_Cuboid):
                        union[res] = None
                    else:
                        for rect in res:
                            union[rect] = None
            except TypeError:
                for rect in self.lst+other:
                    union[rect] = None
            return BoxSet(list(union))
        else:
            if not self.lst:
                return other
            if not other.lst:
                return self
            cuboids = get_cuboids(self)
            new_f2 = recursive_substract(cuboids,other)
            if type(new_f2) != type(None):
                return BoxSet([self,new_f2])
            return self
            
    def __or__(self,other):
        sd = []
        if not self.lst:
            return other
        if not other.lst:
            return self
        if self.lst == other.lst:
            return BoxSet([])
                
        
        rest_of_a = []
        for a in self.lst:
            rest_of_a.append(a)
            for b in other.lst:
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - b)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
        
        sd.extend(rest_of_a)   
        
        
        rest_of_b = []
        for b in other.lst:
            rest_of_b.append(b)
            for a in self.lst:
                toadd = []
                todel = []
                for b_prime in rest_of_b:
                    todel.append(b_prime)
                    toadd.extend(b_prime - a)
                for d in todel:
                    rest_of_b.remove(d)
                rest_of_b.extend(toadd)
        
        sd.extend(rest_of_b)        
        
        return BoxSet(list(set(sd)))



    def __sub__(self,other):
        sub = []
        if not self.lst:
            return []
            
        if isinstance(other,N_Cuboid):
            rest_of_a = []
            for a in self.lst:
                rest_of_a.append(a)
                toadd = []
                todel = []
                for a_prime in rest_of_a:
                    todel.append(a_prime)
                    toadd.extend(a_prime - other)
                for d in todel:
                    rest_of_a.remove(d)
                rest_of_a.extend(toadd)
            sub.extend(rest_of_a)
            return BoxSet(list(set(sub)))
            





def symmetric_difference_distance(phi1,phi2,rand_area,dimensions):
    """
        Function computing the symmetric difference distance between 2 STL Formulae.
        Takes as input:
            * phi1: an STL Formula
            * phi2: an STL Formula
            * rand_area: the domain on which signals are defined. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * dimensions: the dimensions on which the 2 STLFormulae are defined, e.g. dimensions=['x','y'].
        Follows the definition of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision andControl (CDC). pp. 1542–1547
    """
    
    max_horizon = max(phi1.horizon,phi2.horizon)
    dimensions = dimensions+['t']
    
    def area(boxelement):
        if isinstance(boxelement,N_Cuboid):
            return boxelement.n_dim_volume
        if isinstance(boxelement,ChoiceSet):
            if boxelement.lst:
                tmp = 0.0
                for choicebox in boxelement.lst:
                    tmp += area(choicebox)
                return tmp/len(boxelement.lst)
        if isinstance(boxelement,BoxSet):
            tmp = 0.0
            for box in boxelement.lst:
                tmp += area(box)
            return tmp
        return 0.0

    
    def aos(phi,rand_area,max_horizon,dimensions,tab=''):
        # print(tab,"processing",phi)
        if isinstance(phi, STLFormula.TrueF):
            dict_dimensions = {}
            for dimension in dimensions:
                dict_dimensions[dimension] = (0,1)
            dict_dimensions['t'] = (0,max_horizon)
            return N_Cuboid_TrueFalse(dict_dimensions,dimensions)
        elif isinstance(phi, STLFormula.FalseF):
            dict_dimensions = {}
            for dimension in dimensions:
                dict_dimensions[dimension] = (0,0)
            dict_dimensions['t'] = (0,max_horizon)
            return N_Cuboid_TrueFalse(dict_dimensions,dimensions)
        elif isinstance(phi, STLFormula.Predicate):
            if phi.operator == operatorclass.gt or phi.operator == operatorclass.ge:
                dict_dimensions = {}
                for dimension in dimensions:
                    dict_dimensions[dimension] = (0,1)
                dict_dimensions[phi.dimension] = (phi.mu/(rand_area[1]-rand_area[0]),1)
                dict_dimensions['t'] = (0,0)
                return N_Cuboid_Predicate(dict_dimensions,dimensions)
            else:  
                dict_dimensions = {}
                for dimension in dimensions:
                    dict_dimensions[dimension] = (0,1)
                dict_dimensions[phi.dimension] = (0,phi.mu/(rand_area[1]-rand_area[0]))
                dict_dimensions['t'] = (0,0)
                return N_Cuboid_Predicate(dict_dimensions,dimensions)
        elif isinstance(phi, STLFormula.Conjunction):
            f1 = aos(phi.first_formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            f2 = aos(phi.second_formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            f1 = refactor_aos(f1)
            f2 = refactor_aos(f2)
            if type(f1).__name__ == 'bool' and type(f2).__name__ == 'bool':
                return False
            if type(f1).__name__ == 'bool':
                return f2
            if type(f2).__name__ == 'bool':
                return f1
            if isinstance(f1,N_Cuboid_Predicate) and isinstance(f2,N_Cuboid_Predicate):
                return f1 & f2
            elif isinstance(f1,N_Cuboid_TrueFalse) or isinstance(f2,N_Cuboid_TrueFalse):
                return f1 & f2
            elif isinstance(f1,N_Cuboid_Timed) and  isinstance(f2,N_Cuboid_Timed):
                return BoxSet(f1+f2)
            elif isinstance(f1,N_Cuboid) and isinstance(f2,ChoiceSet):
                cs = []
                for c in f2.lst:
                    res = c-f1
                    if isinstance(res,ChoiceSet):
                        cs.append(res)
                    else:
                        cs.extend(c-f1)
                if not cs:
                    return f1
                return BoxSet([f1,ChoiceSet(cs)])
            elif isinstance(f1,ChoiceSet) and isinstance(f2,N_Cuboid):
                cs = []
                for c in f1.lst:
                    res = c-f2
                    if isinstance(res,ChoiceSet):
                        cs.append(res)
                    else:
                        cs.extend(c-f2)
                if not cs:
                    return f2
                return BoxSet([f2,ChoiceSet(cs)])
            elif isinstance(f1,BoxSet) and isinstance(f2,BoxSet):
                return BoxSet(f1+f2)
            elif isinstance(f1,ChoiceSet) and isinstance(f2,ChoiceSet):
                lst = []
                for i,j in product(f1.lst,f2.lst):
                    res = i+j
                    if isinstance(res,BoxSet) or isinstance(res,ChoiceSet):
                        lst.append(res)
                    else:
                        lst.append(BoxSet(res))
                return ChoiceSet(lst)
            elif isinstance(f1,BoxSet) and isinstance(f2,ChoiceSet):
                cuboids = get_cuboids(f1)
                new_f2 = recursive_substract(cuboids,f2)
                if type(new_f2) != type(None):
                    return BoxSet([f1,new_f2])
                return f1
            elif isinstance(f1,ChoiceSet) and isinstance(f2,BoxSet):
                cuboids = get_cuboids(f2)
                new_f1 = recursive_substract(cuboids,f1)
                if type(new_f1) != type(None):
                    return BoxSet([new_f1,f2])
                return f2
            else:
                return BoxSet([f1,f2])
        elif isinstance(phi, STLFormula.Disjunction):
            p1 = aos(phi.first_formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            p2 = aos(phi.second_formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            p1 = refactor_aos(p1)
            p2 = refactor_aos(p2)
            if type(p1).__name__ == 'bool' and type(p2).__name__ == 'bool':
                return False
            if type(p1).__name__ == 'bool':
                return p2
            if type(p2).__name__ == 'bool':
                return p1
            if isinstance(p1,ChoiceSet) and isinstance(p2,ChoiceSet):
                lst = copy.deepcopy(p1.lst)
                lst.extend(p2.lst)
                return ChoiceSet(lst)
            return ChoiceSet([p1,p2])
        elif isinstance(phi, STLFormula.Always):
            f = aos(phi.formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            f = refactor_aos(f)
            if type(f).__name__ == 'bool':
                return False
            if isinstance(f,N_Cuboid_Predicate):
                dict_dimensions = {}
                dict_dimensions = copy.deepcopy(f.dict_dimensions)
                dict_dimensions['t'] = (phi.t1,phi.t2)
                return N_Cuboid_Timed(dict_dimensions,dimensions)
            elif isinstance(f,N_Cuboid_Timed):
                dict_dimensions = {}
                dict_dimensions = copy.deepcopy(f.dict_dimensions)
                dict_dimensions['t'] = (dict_dimensions['t'][0]+phi.t1,dict_dimensions['t'][1]+phi.t2)
                return N_Cuboid_Timed(dict_dimensions,dimensions)
            elif isinstance(f,BoxSet):
                lst = []
                for b in f.lst:
                    try:
                        dict_dimensions = {}
                        dict_dimensions = copy.deepcopy(b.dict_dimensions)
                        dict_dimensions['t'] = (dict_dimensions['t'][0]+phi.t1,dict_dimensions['t'][1]+phi.t2)
                        lst.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                    except AttributeError:
                        continue
                return BoxSet(lst)
            elif isinstance(f,ChoiceSet):
                lst = []
                for b in f.lst:
                    dict_dimensions = {}
                    dict_dimensions = copy.deepcopy(b.dict_dimensions)
                    dict_dimensions['t'] = (dict_dimensions['t'][0]+phi.t1,dict_dimensions['t'][1]+phi.t2)
                    lst.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                return ChoiceSet(lst)
            else:
                print("error, not implemented")
                exit()
        elif isinstance(phi, STLFormula.Eventually):
            cs = []
            f = aos(phi.formula,rand_area,max_horizon,dimensions,tab=tab+'\t')
            f = refactor_aos(f)
            if type(f).__name__ == 'bool':
                return False
            if isinstance(f,N_Cuboid_Predicate):
                for i in range(phi.t1,phi.t2):
                    dict_dimensions = copy.deepcopy(f.dict_dimensions)
                    dict_dimensions['t'] = (i,i+1)
                    cs.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                return ChoiceSet(cs)
            elif isinstance(f,N_Cuboid_Timed):
                for i in range(phi.t1,phi.t2):
                    dict_dimensions = copy.deepcopy(f.dict_dimensions)
                    dict_dimensions['t'] = (dict_dimensions['t'][0]+i,dict_dimensions['t'][1]+i+1)
                    cs.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                return ChoiceSet(cs)
            elif isinstance(f,BoxSet):
                for i in range(phi.t1,phi.t2):
                    lst = []
                    for b in f:
                        dict_dimensions = {}
                        dict_dimensions = copy.deepcopy(b.dict_dimensions)
                        dict_dimensions['t'] = (dict_dimensions['t'][0]+phi.t1+i,dict_dimensions['t'][1]+phi.t2+i)
                        lst.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                    cs.append(BoxSet(lst))
                return ChoiceSet(cs)
            elif isinstance(f,ChoiceSet):
                for i in range(phi.t1,phi.t2):
                    lst = []
                    for b in f.lst:
                        dict_dimensions = {}
                        dict_dimensions = copy.deepcopy(b.dict_dimensions)
                        dict_dimensions['t'] = (dict_dimensions['t'][0]+phi.t1+i,dict_dimensions['t'][1]+phi.t2+i)
                        lst.append(N_Cuboid_Timed(dict_dimensions,dimensions))
                    cs.append(ChoiceSet(lst))
                return ChoiceSet(cs)
            else:
                print("error, not implemented")
                exit()
            
        elif isinstance(phi, STLFormula.Negation):
            #So far, work with the NNF of the STL Formulae
            print("error, not implemented")
            exit()
    
    
    b1 = aos(phi1,rand_area,max_horizon,dimensions)
    b1 = refactor_aos(b1)
    # print("computed b1")
    # print(b1,type(b1).__name__)
    # b1.show()
    
    b2 = aos(phi2,rand_area,max_horizon,dimensions)
    b2 = refactor_aos(b2)
    # print("computed b2")
    # print(b2,type(b2).__name__)
    # b2.show()
    
    
    # print("now calculating sd")
    
    
    def sd(entity1,entity2):
        if isinstance(entity1,N_Cuboid) and isinstance(entity2,N_Cuboid):
            return BoxSet(entity1 | entity2)
        
        elif isinstance(entity1,N_Cuboid) and isinstance(entity2,ChoiceSet):
            cs = []
            for choice in entity2.lst:
                cs.append(sd(entity1, choice))
            return ChoiceSet(list(set(cs)))
        elif isinstance(entity1,ChoiceSet) and isinstance(entity2,N_Cuboid):
            cs = []
            for choice in entity1.lst:
                cs.append(sd(entity2, choice))
            return ChoiceSet(list(set(cs)))
        
        elif isinstance(entity1,N_Cuboid) and isinstance(entity2,BoxSet):
            symd = []
            symd_sets = []
        
            rest_of_a = []
            for a in [entity1]:
                seen_cuboid = False
                rest_of_a.append(a)
                for b in entity2.lst:
                    if isinstance(b,N_Cuboid) or isinstance(b,N_Cuboid_TrueFalse):
                        seen_cuboid = True
                        toadd = []
                        todel = []
                        for a_prime in rest_of_a:
                            todel.append(a_prime)
                            toadd.extend(a_prime - b)
                        for d in todel:
                            rest_of_a.remove(d)
                        rest_of_a.extend(toadd)
                    else:
                        symd_sets.append(sd(entity1,b))
            
            if seen_cuboid:
                symd.extend(rest_of_a)   
            
            rest_of_b = []
            for b in entity2.lst:
                if isinstance(b,N_Cuboid) or isinstance(b,N_Cuboid_TrueFalse):
                    rest_of_b.append(b)
                    for a in [entity1]:
                        toadd = []
                        todel = []
                        for b_prime in rest_of_b:
                            todel.append(b_prime)
                            toadd.extend(b_prime - a)
                        for d in todel:
                            rest_of_b.remove(d)
                        rest_of_b.extend(toadd)
                
            
            symd.extend(rest_of_b) 
                
            res = BoxSet([])
            for s in list(set(symd)):
                res = res + BoxSet([s])

            res.lst.extend(symd_sets)
            
            return res


        elif isinstance(entity1,BoxSet) and isinstance(entity2,N_Cuboid):
        
            symd = []
            symd_sets = []
        
            rest_of_a = []
            for a in [entity2]:
                rest_of_a.append(a)
                for b in entity1.lst:
                    if isinstance(b,N_Cuboid) or isinstance(b,N_Cuboid_TrueFalse):
                        toadd = []
                        todel = []
                        for a_prime in rest_of_a:
                            todel.append(a_prime)
                            toadd.extend(a_prime - b)
                        for d in todel:
                            rest_of_a.remove(d)
                        rest_of_a.extend(toadd)
                    else:
                        symd_sets.append(sd(entity2,b))
            
            symd.extend(rest_of_a)   
            
            rest_of_b = []
            for b in entity1.lst:
                if isinstance(b,N_Cuboid) or isinstance(b,N_Cuboid_TrueFalse):
                    rest_of_b.append(b)
                    for a in [entity2]:
                        toadd = []
                        todel = []
                        for b_prime in rest_of_b:
                            todel.append(b_prime)
                            toadd.extend(b_prime - a)
                        for d in todel:
                            rest_of_b.remove(d)
                        rest_of_b.extend(toadd)
                
            
            symd.extend(rest_of_b) 


                
            res = BoxSet([])
            for s in list(set(symd)):
                res = res + BoxSet([s])

            res.lst.extend(symd_sets)

            return res
            
            
            
            
        elif isinstance(entity1,ChoiceSet) and isinstance(entity2,ChoiceSet):            
            cs1 = []
            count = 1
            for e1 in entity1.lst:
                best_sd = []
                lowest_area = float("inf")
                for e2 in entity2.lst:
                    sd_elts = sd(e1,e2)
                    if area(sd_elts) < lowest_area:
                        best_sd = sd_elts
                        lowest_area = area(sd_elts)
                    if lowest_area == 0.0:
                        break
                cs1.append(best_sd)
                count += 1
            return ChoiceSet(list(set(cs1)))
        
        elif isinstance(entity1,ChoiceSet) and isinstance(entity2,BoxSet):
            boxsets = []
            for choice in entity1.lst:
                boxsets.append(sd(choice,entity2))
            return ChoiceSet(boxsets)
        elif isinstance(entity1,BoxSet) and isinstance(entity2,ChoiceSet):
            boxsets = []
            for choice in entity2.lst:
                boxsets.append(sd(choice,entity1))
            return ChoiceSet(boxsets)
            
        elif isinstance(entity1,BoxSet) and isinstance(entity2,BoxSet):
            boxsets = BoxSet([])
            for box in entity1.lst:
                boxsets = sd(boxsets,box)
            for box in entity2.lst:
                boxsets = sd(boxsets,box)
            return boxsets
        
        else:
            print("SD not implemented")
            exit()
            
    sdiff = sd(b1,b2)
    sdiff = refactor_aos(sdiff)
    # sdiff.show()
        
    return round(area(sdiff)/max_horizon,5)









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
    phi4 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20),STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20))
    phi5 = STLFormula.Conjunction(STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,10),STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),12,20))
    phi6 = STLFormula.Always(STLFormula.Eventually(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,4),0,16)
    true = STLFormula.TrueF()


    #Reproduces the results of Madsen et al. (2018)
    rand_area = [0,1]
    dimensions = ['x']
    
    #Symetric Difference    
    print('sd(True,phi1)',symmetric_difference_distance(true,phi1,rand_area,dimensions),'(0.8)')
    print('sd(phi1,phi1)',symmetric_difference_distance(phi1,phi1,rand_area,dimensions),'(0.0)') 
    print('sd(phi1,phi2)',symmetric_difference_distance(phi1,phi2,rand_area,dimensions),'(0.04)')
    print('sd(phi1,phi3)',symmetric_difference_distance(phi1,phi3,rand_area,dimensions),'(0.19)')
    print('sd(phi1,phi4)',symmetric_difference_distance(phi1,phi4,rand_area,dimensions),'(0.0)')
    print('sd(phi1,phi5)',symmetric_difference_distance(phi1,phi5,rand_area,dimensions),'(0.036)')
    print('sd(phi1,phi6)',symmetric_difference_distance(phi1,phi6,rand_area,dimensions),'(0.03)')

    print('')

    print('sd(True,phi2)',symmetric_difference_distance(true,phi2,rand_area,dimensions),'(0.76)')
    print('sd(phi2,phi2)',symmetric_difference_distance(phi2,phi2,rand_area,dimensions),'(0.0)')
    print('sd(phi2,phi3)',symmetric_difference_distance(phi2,phi3,rand_area,dimensions),'(0.23)')
    print('sd(phi2,phi4)',symmetric_difference_distance(phi2,phi4,rand_area,dimensions),'(0.04)')
    print('sd(phi2,phi5)',symmetric_difference_distance(phi2,phi5,rand_area,dimensions),'(0.044)')
    print('sd(phi2,phi6)',symmetric_difference_distance(phi2,phi6,rand_area,dimensions),'(0.07)')

    print('')

    print('sd(True,phi3)',symmetric_difference_distance(true,phi3,rand_area,dimensions),'(0.99)')
    print('sd(phi3,phi3)',symmetric_difference_distance(phi3,phi3,rand_area,dimensions),'(0.0)')
    print('sd(phi3,phi4)',symmetric_difference_distance(phi3,phi4,rand_area,dimensions),'(0.19)')
    print('sd(phi3,phi5)',symmetric_difference_distance(phi3,phi5,rand_area,dimensions),'(0.188)')
    print('sd(phi3,phi6)',symmetric_difference_distance(phi3,phi6,rand_area,dimensions),'(0.16)')
    print('')

    print('sd(True,phi4)',symmetric_difference_distance(true,phi4,rand_area,dimensions),'(0.8)')
    print('sd(phi4,phi4)',symmetric_difference_distance(phi4,phi4,rand_area,dimensions),'(0.0)')
    print('sd(phi4,phi5)',symmetric_difference_distance(phi4,phi5,rand_area,dimensions),'(0.036)')
    print('sd(phi4,phi6)',symmetric_difference_distance(phi4,phi6,rand_area,dimensions),'(0.03)')

    print('')

    print('sd(True,phi5)',symmetric_difference_distance(true,phi5,rand_area,dimensions),'(0.804)')
    print('sd(phi5,phi5)',symmetric_difference_distance(phi5,phi5,rand_area,dimensions),'(0.0)')
    print('sd(phi5,phi6)',symmetric_difference_distance(phi5,phi6,rand_area,dimensions),'(0.066)')

    print('')

    print('sd(True,phi6)',symmetric_difference_distance(true,phi6,rand_area,dimensions),'(0.83)')
    print('sd(phi6,phi6)',symmetric_difference_distance(phi6,phi6,rand_area,dimensions),'(0.0)')    
    
    
    print('')
    print('')
    
    
    
    # #Pompeiu-Hausdorff
    print('ph(True,phi1)',pompeiu_hausdorff_distance(true,phi1,rand_area,dimensions),'(0.6)')
    print('ph(phi1,phi1)',pompeiu_hausdorff_distance(phi1,phi1,rand_area,dimensions),'(0.0)')
    print('ph(phi1,phi2)',pompeiu_hausdorff_distance(phi1,phi2,rand_area,dimensions),'(0.04)')
    print('ph(phi1,phi3)',pompeiu_hausdorff_distance(phi1,phi3,rand_area,dimensions),'(0.6)')
    print('ph(phi1,phi4)',pompeiu_hausdorff_distance(phi1,phi4,rand_area,dimensions),'(0.0)')
    print('ph(phi1,phi5)',pompeiu_hausdorff_distance(phi1,phi5,rand_area,dimensions),'(0.6)')
    print('ph(phi1,phi6)',pompeiu_hausdorff_distance(phi1,phi6,rand_area,dimensions),'(0.6)')

    print('')

    print('ph(True,phi2)',pompeiu_hausdorff_distance(true,phi2,rand_area,dimensions),'(0.56)')
    print('ph(phi2,phi2)',pompeiu_hausdorff_distance(phi2,phi2,rand_area,dimensions),'(0.0)')
    print('ph(phi2,phi3)',pompeiu_hausdorff_distance(phi2,phi3,rand_area,dimensions),'(0.56)')
    print('ph(phi2,phi4)',pompeiu_hausdorff_distance(phi2,phi4,rand_area,dimensions),'(0.04)')
    print('ph(phi2,phi5)',pompeiu_hausdorff_distance(phi2,phi5,rand_area,dimensions),'(0.56)')
    print('ph(phi2,phi6)',pompeiu_hausdorff_distance(phi2,phi6,rand_area,dimensions),'(0.56)')

    print('')

    print('ph(True,phi3)',pompeiu_hausdorff_distance(true,phi3,rand_area,dimensions),'(0.6)')
    print('ph(phi3,phi3)',pompeiu_hausdorff_distance(phi3,phi3,rand_area,dimensions),'(0.0)')
    print('ph(phi3,phi4)',pompeiu_hausdorff_distance(phi3,phi4,rand_area,dimensions),'(0.6)')
    print('ph(phi3,phi5)',pompeiu_hausdorff_distance(phi3,phi5,rand_area,dimensions),'(0.6)')
    print('ph(phi3,phi6)',pompeiu_hausdorff_distance(phi3,phi6,rand_area,dimensions),'(0.6)')

    print('')

    print('ph(True,phi4)',pompeiu_hausdorff_distance(true,phi4,rand_area,dimensions),'(0.6)')
    print('ph(phi4,phi4)',pompeiu_hausdorff_distance(phi4,phi4,rand_area,dimensions),'(0.0)')
    print('ph(phi4,phi5)',pompeiu_hausdorff_distance(phi4,phi5,rand_area,dimensions),'(0.6)')
    print('ph(phi4,phi6)',pompeiu_hausdorff_distance(phi4,phi6,rand_area,dimensions),'(0.6)')

    print('')

    print('ph(True,phi5)',pompeiu_hausdorff_distance(true,phi5,rand_area,dimensions),'(0.6)')
    print('ph(phi5,phi5)',pompeiu_hausdorff_distance(phi5,phi5,rand_area,dimensions),'(0.0)')
    print('ph(phi5,phi6)',pompeiu_hausdorff_distance(phi5,phi6,rand_area,dimensions),'(0.6)')

    print('')

    print('ph(True,phi6)',pompeiu_hausdorff_distance(true,phi6,rand_area,dimensions),'(0.6)')
    print('ph(phi6,phi6)',pompeiu_hausdorff_distance(phi6,phi6,rand_area,dimensions),'(0.0)')
    
    
    print('')
    print('\n\n\n\n\n\n\n\n\n')
    
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
    always_nnf = STLFormula.toNegationNormalForm(always,False)

    phi_hypothesis = STLFormula.Conjunction(eventually1,eventually2)

    rand_area = [0,4]
    dimensions = ['x','y']

    print(pompeiu_hausdorff_distance(true,phi_target,rand_area,dimensions),symmetric_difference_distance(true,phi_target,rand_area,dimensions))
    print(pompeiu_hausdorff_distance(true,phi_target,rand_area,dimensions),symmetric_difference_distance(true,STLFormula.Conjunction(eventually1,always_nnf),rand_area,dimensions))
    print(pompeiu_hausdorff_distance(eventually1,eventually2,rand_area,dimensions),symmetric_difference_distance(eventually1,eventually2,rand_area,dimensions))

    print('')
    print('')

    predicate_x_gt0 = STLFormula.Predicate('x',operatorclass.gt,0,INDEX_X)
    predicate_x_le1 = STLFormula.Predicate('x',operatorclass.le,1,INDEX_X)
    predicate_y_gt3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
    predicate_y_le4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    always_1 = STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt0,predicate_x_le1) , STLFormula.Conjunction(predicate_y_gt3,predicate_y_le4) ),10,20)
    predicate_x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,INDEX_X)
    predicate_x_le4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt0 = STLFormula.Predicate('y',operatorclass.gt,0,INDEX_Y)
    predicate_y_le1 = STLFormula.Predicate('y',operatorclass.le,1,INDEX_Y)
    always_2 = STLFormula.Always(STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt3,predicate_x_le4) , STLFormula.Conjunction(predicate_y_gt0,predicate_y_le1) ),30,50)
    predicate_x_gt2 = STLFormula.Predicate('x',operatorclass.gt,2,INDEX_X)
    predicate_y_gt1 = STLFormula.Predicate('y',operatorclass.gt,1,INDEX_Y)
    predicate_x_le3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
    predicate_y_le2 = STLFormula.Predicate('y',operatorclass.le,2,INDEX_Y)
    always_not = STLFormula.toNegationNormalForm(STLFormula.Always( STLFormula.Negation( STLFormula.Conjunction( STLFormula.Conjunction(predicate_x_gt2,predicate_x_le3) , STLFormula.Conjunction(predicate_y_gt1,predicate_y_le2) ) ),0,50),False) 
    
    phi = STLFormula.Conjunction(STLFormula.Conjunction(always_1,always_2),always_not)

    rand_area = [0,4]
    dimensions = ['x','y']
    
    print(pompeiu_hausdorff_distance(STLFormula.Conjunction(always_1,always_2),always_not,rand_area,dimensions),symmetric_difference_distance(STLFormula.Conjunction(always_1,always_2),always_not,rand_area,dimensions))
    print(pompeiu_hausdorff_distance(STLFormula.Conjunction(always_1,always_2),STLFormula.Conjunction(STLFormula.Conjunction(always_1,always_2),always_not),rand_area,dimensions),symmetric_difference_distance(STLFormula.Conjunction(always_1,always_2),STLFormula.Conjunction(STLFormula.Conjunction(always_1,always_2),always_not),rand_area,dimensions))