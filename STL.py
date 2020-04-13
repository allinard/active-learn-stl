import operator as operatorclass
import random

#Definition of operators
operators_iv = {
    operatorclass.gt: ">",
    operatorclass.ge: "\geq",
    operatorclass.lt: "<",
    operatorclass.le: "\leq"
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
    Class for representing an STL Formula.
    """
    
    
    
    def toNegationNormalForm(phi,inNegation):
        """
        Recursive function returning the Negation Normal Form of an STL Formula
        """
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
        """
        Function simplifying an STL Formula
        """
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
    
    
    

    
    
    
    class TrueF:
        """
        Class representing the True boolean constant
        """
        def __init__(self):
            self.robustness = lambda s, t : float('inf')
            self.horizon = 0
            
        def __str__(self):
            return "\\top"



    class FalseF:
        """
        Class representing the False boolean constant
        """
        def __init__(self):
            self.robustness = lambda s, t : float('-inf')
            self.horizon = 0
            
        def __str__(self):
            return "\\bot"



    class Predicate:
        """
        Class representing a Predicate, s.t. f(s) \sim \mu
        The constructor takes 4 arguments:
            * dimension: string/name of the dimension
            * operator: operator (geq, lt...)
            * mu: \mu
            * pi_index_signal: in the signal, which index corresponds to the predicate's dimension
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
            * horizon: 0
        """
        def __init__(self,dimension,operator,mu,pi_index_signal):
            self.pi_index_signal = pi_index_signal
            self.dimension = dimension
            self.operator = operator
            self.mu = mu
            if operator == operatorclass.gt or operator == operatorclass.ge:
                self.robustness = lambda s, t : s[t][pi_index_signal] - mu
            else:
                self.robustness = lambda s, t : -s[t][pi_index_signal] + mu
            
            self.horizon = 0
        
        def __str__(self):
            return self.dimension+operators_iv[self.operator]+str(self.mu)



    class Conjunction: 
        """
        Class representing the Conjunction operator, s.t. \phi_1 \land \phi_2.
        The constructor takes 2 arguments:
            * formula 1: \phi_1
            * formula 2: \phi_2
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\phi_1 \land \phi_2,t) = \min(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
            * horizon: \left\|\phi_1 \land \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
        """
        def __init__(self,first_formula,second_formula):
            self.first_formula = first_formula
            self.second_formula = second_formula
            self.robustness = lambda s, t : min( first_formula.robustness(s,t),
                                            second_formula.robustness(s,t) )
            self.horizon = max( first_formula.horizon, 
                                            second_formula.horizon )
        
        def __str__(self):
            return "("+str(self.first_formula)+") \wedge ("+str(self.second_formula)+")"



    class Negation: 
        """
        Class representing the Negation operator, s.t. \neg \phi.
        The constructor takes 1 argument:
            * formula 1: \phi
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\neg \phi,t) = - \rho(s,\phi,t)
            * horizon: \left\|\phi\right\|=\left\|\neg \phi\right\|
        """
        def __init__(self,formula):
            self.formula = formula
            self.robustness = lambda s, t : -formula.robustness(s,t)
            self.horizon = formula.horizon
        
        def __str__(self):
            return "\lnot ("+str(self.formula)+")"

    
    
    class Disjunction: 
        """
        Class representing the Disjunction operator, s.t. \phi_1 \vee \phi_2.
        The constructor takes 2 arguments:
            * formula 1: \phi_1
            * formula 2: \phi_2
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\phi_1 \lor \phi_2,t) = \max(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
            * horizon: \left\|\phi_1 \lor \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
        """
        def __init__(self,first_formula,second_formula):
            self.first_formula = first_formula
            self.second_formula = second_formula
            self.robustness = lambda s, t : max( first_formula.robustness(s,t),
                                            second_formula.robustness(s,t) )
            self.horizon = max( first_formula.horizon, 
                                            second_formula.horizon )
        
        def __str__(self):
            return "("+str(self.first_formula)+") \\vee ("+str(self.second_formula)+")"

    
    
    class Always: 
        """
        Class representing the Always operator, s.t. \mathcal{G}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{G}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\min~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{G}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t : min([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"



    class Eventually: 
        """
        Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{F}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\max~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{F}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"
    
    
    
    def Xor(first_formula,second_formula):
        """
        Function returning the XOR of 2 STL formulae:
        
        \phi_1 \oplus \phi_2 = (\phi_1 \wedge \neg \phi_2) \vee (\neg \phi_1 \wedge \phi_2)
        """
        if isinstance(first_formula, STLFormula.TrueF):
            # return STLFormula.Negation(second_formula)
            # trick for active learning algorithm: when the hypothesis is True, it is helpful to generate positive signals !
            return second_formula
        # trick for faster signal generation of a XOR: randomly return either
        if random.random() > 0.5:
            return STLFormula.Conjunction(first_formula,STLFormula.Negation(second_formula))
        else: 
            return STLFormula.Conjunction(STLFormula.Negation(first_formula),second_formula)
