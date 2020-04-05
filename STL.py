import numpy as np
import operator as operatorclass
import intervals as I
import random
import sys

operators = {
    ">": operatorclass.gt,
    ">=": operatorclass.ge,
    "<": operatorclass.lt,
    "<=": operatorclass.le
}

operators_iv = {
    operatorclass.gt: ">",
    operatorclass.ge: "\geq",
    operatorclass.lt: "<",
    operatorclass.le: "\leq"
}

operators_iv_g = {
    operatorclass.gt: "_gt_",
    operatorclass.ge: "_ge_",
    operatorclass.lt: "_lt_",
    operatorclass.le: "_le_"
}

negation_operators = {
    operatorclass.gt: operatorclass.le,
    operatorclass.ge: operatorclass.lt,
    operatorclass.lt: operatorclass.ge,
    operatorclass.le: operatorclass.gt,
    operatorclass.eq: operatorclass.ne,
    operatorclass.ne: operatorclass.eq
}



class STLFormula:
    """
    An STL Formula. It consists of
        - A robustness function (robustness)
        - An Abstract Syntax Tree (ast)
    """
    
    
    
    def toNegationNormalForm(phi,inNegation):
        if isinstance(phi, STLFormula.TrueF):
            if inNegation:
                return STLFormula.FalseF()
            return phi
        if isinstance(phi, STLFormula.FalseF):
            if inNegation:
                return STLFormula.TrueF()
            return phi
        if isinstance(phi, STLFormula.Negation):
            if inNegation:
                return STLFormula.toNegationNormalForm(phi.formula,False)
            return STLFormula.toNegationNormalForm(phi.formula,True)
        if isinstance(phi, STLFormula.Eventually):
            if inNegation:
                return STLFormula.Always(STLFormula.toNegationNormalForm(STLFormula.Negation(phi.formula),False),phi.t1,phi.t2)
            return STLFormula.Eventually(STLFormula.toNegationNormalForm(phi.formula,inNegation),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Always):
            if inNegation:
                return STLFormula.Eventually(STLFormula.toNegationNormalForm(STLFormula.Negation(phi.formula),False),phi.t1,phi.t2)
            return STLFormula.Always(STLFormula.toNegationNormalForm(phi.formula,inNegation),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Disjunction):
            if inNegation:
                return STLFormula.Conjunction(STLFormula.toNegationNormalForm(phi.first_formula,inNegation),STLFormula.toNegationNormalForm(phi.second_formula,inNegation))
            return STLFormula.Disjunction(STLFormula.toNegationNormalForm(phi.first_formula,inNegation),STLFormula.toNegationNormalForm(phi.second_formula,inNegation))
        if isinstance(phi, STLFormula.Conjunction):
            if inNegation:
                return STLFormula.Disjunction(STLFormula.toNegationNormalForm(phi.first_formula,inNegation),STLFormula.toNegationNormalForm(phi.second_formula,inNegation))
            return STLFormula.Conjunction(STLFormula.toNegationNormalForm(phi.first_formula,inNegation),STLFormula.toNegationNormalForm(phi.second_formula,inNegation))
        if isinstance(phi, STLFormula.Predicate):
            if inNegation:
                return STLFormula.Predicate(phi.dimension,negation_operators[phi.operator],phi.mu,phi.pi_index_signal)
            return phi
    
    
    
    
    def simplify_dtlearn(phi):
        if isinstance(phi, STLFormula.TrueF):
            return phi
        if isinstance(phi, STLFormula.FalseF):
            return phi
        if isinstance(phi, STLFormula.Negation):
            return STLFormula.Negation(STLFormula.simplify_dtlearn(phi.formula))
        if isinstance(phi, STLFormula.Eventually):
            return STLFormula.Eventually(STLFormula.simplify_dtlearn(phi.formula),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Always):
            return STLFormula.Always(STLFormula.simplify_dtlearn(phi.formula),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Disjunction):
            left = STLFormula.simplify_dtlearn(phi.first_formula)
            right = STLFormula.simplify_dtlearn(phi.second_formula)
            if isinstance(left, STLFormula.TrueF) or isinstance(right, STLFormula.TrueF):
                return STLFormula.TrueF()
            elif isinstance(left, STLFormula.FalseF):
                if isinstance(right, STLFormula.FalseF):
                    return STLFormula.FalseF()
                return right
            elif isinstance(right, STLFormula.FalseF):
                return left
            else:
                return STLFormula.Disjunction(left,right)
        if isinstance(phi, STLFormula.Conjunction):
            left = STLFormula.simplify_dtlearn(phi.first_formula)
            right = STLFormula.simplify_dtlearn(phi.second_formula)
            if isinstance(left, STLFormula.FalseF) or isinstance(right, STLFormula.FalseF):
                return STLFormula.FalseF()
            elif isinstance(left, STLFormula.TrueF):
                if isinstance(right, STLFormula.TrueF):
                    return STLFormula.TrueF()
                else:
                    return right
            elif isinstance(right,STLFormula.TrueF):
                return left
            else:
                return STLFormula.Conjunction(left,right)
        if isinstance(phi, STLFormula.Predicate):
            return phi
    
    
    
    def simplify(phi):
        if isinstance(phi, STLFormula.TrueF):
            return phi
        if isinstance(phi, STLFormula.FalseF):
            return phi
        if isinstance(phi, STLFormula.Negation):
            return STLFormula.Negation(STLFormula.simplify(phi.formula))
        if isinstance(phi, STLFormula.Eventually):
            return STLFormula.Eventually(STLFormula.simplify(phi.formula),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Always):
            return STLFormula.Always(STLFormula.simplify(phi.formula),phi.t1,phi.t2)
        if isinstance(phi, STLFormula.Disjunction):
            if random.random() > 0.75:
                if random.random() > 0.5:
                    return STLFormula.simplify(phi.first_formula)
                else:
                    return STLFormula.simplify(phi.second_formula)
            else:
                return STLFormula.Disjunction(STLFormula.simplify(phi.first_formula),STLFormula.simplify(phi.second_formula))
        if isinstance(phi, STLFormula.Conjunction):
            return STLFormula.Conjunction(STLFormula.simplify(phi.first_formula),STLFormula.simplify(phi.second_formula))
        if isinstance(phi, STLFormula.Predicate):
            return phi



    def minI(I1,I2):
        return I.closed( min(I1.lower,I2.lower) , min(I1.upper,I2.upper) )



    def maxI(I1,I2):
        return I.closed( max(I1.lower,I2.lower) , max(I1.upper,I2.upper) )



    def minusI(I1):
        return I.closed(-I1.upper,-I1.lower)



    def minI_l(listI):
        return I.closed( min([i.lower for i in listI]), min([i.upper for i in listI]) )



    def maxI_l(listI):
        return I.closed( max([i.lower for i in listI]), max([i.upper for i in listI]) )
    


    def I_is_singleton(i):
        return i.lower == i.upper
    
    
    
    
    class TrueF:
        def __init__(self):
            self.robustness = lambda s, t : float('inf')
            self.horizon = 1
            
        def tex(self):
            return "\\top"
            
        def __str__(self):
            return "_top"
            
            
    class FalseF:
        def __init__(self):
            self.robustness = lambda s, t : float('-inf')
            self.horizon = 1
            
        def tex(self):
            return "\\bot"
            
        def __str__(self):
            return "_bot"
            
    
    class Predicate:
        def __init__(self,dimension,operator,mu,pi_index_signal):
            self.pi_index_signal = pi_index_signal
            self.dimension = dimension
            self.operator = operator
            self.mu = mu
            if operator == operatorclass.gt or operator == operatorclass.ge:
                self.robustness = lambda s, t : s[t][pi_index_signal] - mu
                # self.robustness = lambda s, t : s[t,pi_index_signal] - mu
            else:
                self.robustness = lambda s, t : -s[t][pi_index_signal] + mu
                # self.robustness = lambda s, t : -s[t,pi_index_signal] + mu
                
            def rosi(s,t):
                try:
                    if s:
                        return I.closed(self.robustness(s,t),self.robustness(s,t))
                    else:
                        return I.open(-I.inf,I.inf)
                except IndexError:
                    return I.open(-I.inf,I.inf)
            self.rosi = rosi
            
            self.horizon = 0
        
        def tex(self):
            return self.dimension+operators_iv[self.operator]+str(self.mu)
            
        def __str__(self):
            return self.dimension+operators_iv_g[self.operator]+str(self.mu)
            
    
    
    class Conjunction: 
        def __init__(self,first_formula,second_formula):
            self.first_formula = first_formula
            self.second_formula = second_formula
            self.robustness = lambda s, t : min( first_formula.robustness(s,t),
                                            second_formula.robustness(s,t) )
            self.horizon = max( first_formula.horizon, 
                                            second_formula.horizon )
            self.rosi = lambda s, t : STLFormula.minI( first_formula.rosi(s,t),
                                            second_formula.rosi(s,t) )
        
        def tex(self):
            return "("+self.first_formula.tex()+") \wedge ("+self.second_formula.tex()+")"
        
        def __str__(self):
            return str(self.first_formula)+"_wedge_"+str(self.second_formula)
        
        
    class Negation: 
        def __init__(self,formula):
            self.formula = formula
            self.robustness = lambda s, t : -formula.robustness(s,t)
            self.horizon = formula.horizon
            self.rosi = lambda s, t : STLFormula.minusI(formula.rosi(s,t))
        
        def tex(self):
            return "\lnot ("+self.formula.tex()+")"
            
        def __str__(self):
            return "_lnot_"+str(self.formula)+"_"
    
    
    class Disjunction: 
        def __init__(self,first_formula,second_formula):
            self.first_formula = first_formula
            self.second_formula = second_formula
            self.robustness = lambda s, t : max( first_formula.robustness(s,t),
                                            second_formula.robustness(s,t) )
            self.horizon = max( first_formula.horizon, 
                                            second_formula.horizon )
            self.rosi = lambda s, t : STLFormula.maxI( first_formula.rosi(s,t),
                                            second_formula.rosi(s,t) )
        
        def tex(self):
            return "("+self.first_formula.tex()+") \\vee ("+self.second_formula.tex()+")"
            
        def __str__(self):
            return str(self.first_formula)+"_vee_"+str(self.second_formula)
    
    
    class Always: 
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t : min([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.rosi = lambda s, t : STLFormula.minI_l([ formula.rosi(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def tex(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+self.formula.tex()+")"
            
        def __str__(self):
            # return "\square_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"
            return "_mathcal_G_"+str(self.t1)+"_"+str(self.t2)+"_"+str(self.formula)+"_"
               

        
    class Eventually: 
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.rosi = lambda s, t :  STLFormula.maxI_l([ formula.rosi(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def tex(self):
            return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+self.formula.tex()+")"
            
        def __str__(self):
            # return "\lozenge_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"
            return "_mathcal_F_"+str(self.t1)+"_"+str(self.t2)+"_"+str(self.formula)+"_"
    
    
    # class Xor: 
        # def __init__(self,first_formula,second_formula):
            # self.first_formula = first_formula
            # self.second_formula = second_formula
            # self.robustness = lambda s, t : max( min( first_formula.robustness(s,t),
                                            # -second_formula.robustness(s,t) ) ,
                                            # min( -first_formula.robustness(s,t),
                                            # second_formula.robustness(s,t) ) )
            # self.rosi = lambda s, t : STLFormula.maxI( STLFormula.minI( first_formula.rosi(s,t),
                                            # -second_formula.rosi(s,t) ) ,
                                            # STLFormula.minI( -first_formula.rosi(s,t),
                                            # second_formula.rosi(s,t) ) )
            # self.horizon = max( first_formula.horizon, 
                                            # second_formula.horizon )  
            
        # def __str__(self):
            # return str(self.first_formula)+" \oplus "+str(self.second_formula)
    
    def Xor(first_formula,second_formula):
        # if isinstance(first_formula, STLFormula.TrueF):
            # return  STLFormula.toNegationNormalForm(STLFormula.Conjunction(second_formula,STLFormula.Negation(second_formula)),False)
        # return STLFormula.toNegationNormalForm(STLFormula.Conjunction(STLFormula.Disjunction(first_formula,second_formula),STLFormula.Disjunction(STLFormula.Negation(first_formula),STLFormula.Negation(second_formula))),False)
        return STLFormula.toNegationNormalForm(second_formula,False)