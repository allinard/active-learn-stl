Active Learning of Signal Temporal Logic Specifications
=============================================

active-learn-stl is a Python library that infers Signal Temporal Logic Specifications, using Active Learning



## Introduction

STL is a temporal logic defined on continuous signals that can be used for specifications on systems. 
It can express system properties that include time bounds and bounds on physical system parameters.
One of its advantage is its richness to specify continuous-time systems.
It is, in the case of robotics, an attractive formalism to model, for instance, classes of desired trajectories.





## Downloading sources

You can use this API by cloning this repository:
```
$ git clone https://github.com/allinard/active-learn-stl.git
```

Dependencies:
* Python 3
	* Pulp
	* Sympy
* Gurobi MILP Solver





## STL

The module `STL.py` implements the formalism of STL Formulae.
It supports several boolean (Conjunction, Disjunction, Negation) and temporal operators (Always, Eventually).


### True and False boolean constants

```
t = STLFormula.TrueF()
f = STLFormula.FalseF()
```


### Predicate

```
x_gt3 = STLFormula.Predicate(dimension,operator,mu,pi_index_signal)
```
is an STL Formula, where the constructor takes 4 arguments:
* `dimension`: string/name of the dimension (ex: `'x'`)
* `operator`: operator (`operatorclass.geq`, `operatorclass.lt`, `operatorclass.leq`, `operatorclass.gt`)
* `mu`: float mu (ex: `3`)
* `pi_index_signal`: in the signal, which index corresponds to the predicate's dimension (ex: `0`)


### Conjunction and Disjunction

```
c = STLFormula.Conjunction(phi1,phi2)
d = STLFormula.Disjunction(phi1,phi2)
```
are STL Formulae respectively representing the Conjunction and Disjunction of 2 STL Formulae `phi1` and `phi2`.


### Negation

```
n = STLFormula.Negation(phi)
```
is an STL Formula representing the negation of an STL Formula `phi`.


### Always and Eventually

```
a = STLFormula.Always(phi,t1,t2)
e = STLFormula.Eventually(phi,t1,t2)
```
are STL Formulae respectively representing the Always and Eventually of an STL Formulae `phi`. They both takes 3 arguments:
* `phi`: an STL formula
* `t1`: lower time interval bound
* `t2`: upper time interval bound


### Robustness

All STL Formulae contain 1 function to compute the robustness of a signal given the STL Formula.

```
x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,0)
a = STLFormula.Always(x_gt3,0,5)
a.robustness([[3.1],[3.3],[3.2],[3.0],[2.9],[3.1],[3.5],[3.1],[2.2]],0)
-0.1
```



## STL Distance

The module `STLDistance.py` implements the calculation of the distance between 2 STL Formulae.

Follows the definitions of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision and Control (CDC). pp. 1542–1547

```
rand_area = [0,1]

x_ge02 = STLFormula.Predicate('x',operatorclass.ge,0.2,INDEX_X)
x_lt04 = STLFormula.Predicate('x',operatorclass.le,0.4,INDEX_X)
x_lt044 = STLFormula.Predicate('x',operatorclass.le,0.44,INDEX_X)
phi1 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
phi2 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20)

pompeiu_hausdorff_distance(phi1,phi2,rand_area)
0.04

symmetric_difference_distance(phi1,phi3,rand_area)
0.19
```
where `pompeiu_hausdorff_distance` and `symmetric_difference_distance` takes as input:
* `phi1`: an STL Formula
* `phi2`: an STL Formula
* `rand_area`: the domain on which signals are generated. `rand_area = [lb,ub]` where `lb` is the lower bound and `ub` the upper bound of the domain.



## STL Generate Signals

The module `STLGenerateSignal.py` implements the generation of signals satisfying an STL Formula.

Follows the definitions of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.


### Boolean Enconding

````
generate_signal_milp_boolean(phi,start,rand_area,U,epsilon)  
````
generates a signal satisfying an STL Formula using boolean encoding of MILP constraints. It takes as input:
* `phi`: an STL Formula
* `start`: a vector of the form `[x0,y0]` for the starting point coordinates
* `rand_area`: the domain on which signals are generated. `rand_area = [lb,ub]` where `lb` is the lower bound and `ub` the upper bound of the domain
* `U`: a basic control policy standing for the max amplitude of the signal between 2 time stamps
* `epsilon`: basic control policy parameter

### Quantitative Enconding

````
generate_signal_milp_quantitative(phi,start,rand_area,U,epsilon,OPTIMIZE_ROBUSTNESS)  
````
generates a signal satisfying an STL Formula using boolean encoding of MILP constraints. It takes as input:
* `phi`: an STL Formula
* `start`: a vector of the form `[x0,y0]` for the starting point coordinates
* `rand_area`: the domain on which signals are generated. `rand_area = [lb,ub]` where `lb` is the lower bound and `ub` the upper bound of the domain
* `U`: a basic control policy standing for the max amplitude of the signal between 2 time stamps
* `epsilon`: basic control policy parameter
* `OPTIMIZE_ROBUSTNESS`: a flag whether the robustness of the generated signal w.r.t. phi has to be maximized or not





## STL DT Learn

The module `STLDTLearn.py` implements the learning of a decision tree for STL Inference.

Follows G. Bombara and C. Belta, "Online learning of temporal logic formulae for signal classification", in 2018 European Control Conference (ECC). IEEE, 2018, pp. 2057–2062

```
dtlearn = DTLearn(rand_area,max_horizon,primitives='MOTION_PLANNING')
```
which initializes a decision-tree for the online inference of an STL Formula.
* `rand_area`: the domain on which signals are generated. `rand_area = [lb,ub]` where `lb` is the lower bound and `ub` the upper bound of the domain.
* `max_horizon`: the maximum horizon of the STL Formula to learn
* `primitives` (optional): either `'MOTION_PLANNING'` or `'CLASSICAL'` (default set to `'CLASSICAL'`). The *classical primitives* are the first-order primitives as defined in Bombara and Belta (2018), and the *motion planning primitives* as defined in *publication under review*.

```
dtlearn.update(signal,label)
```
which updates the decision tree given a labelled signal.


## STL Active Learn

The module `STLActiveLearn.py` implements the active learning framework for the inference of STL Formula.

Follows *publication under review*.

```
active_learn = STLActiveLearn(phi_target,
			      rand_area,
			      start,
			      max_horizon,
			      primitives='MOTION_PLANNING',
			      signal_gen='QUANTITATIVE_OPTIMIZE',
			      U=0.2,
			      epsilon=0.05,
			      alpha=0.01,
			      beta=0.5,
			      gamma=50,
			      MAX_IT=100,
			      phi_hypothesis=STLFormula.TrueF(),
			      plot_activated=True)
active_learn.dtlearn.simple_boolean()
```
which actively learns a candidate STL specification given a System Under Test from which we can query information on its target specification `phi_target`:
* `phi_target`: the target specification to learn
* `rand_area`: the domain on which signals are generated. `rand_area = [lb,ub]` where `lb` is the lower bound and `ub` the upper bound of the domain.
* `start`: a vector of the form `[x0,y0]` for the starting point coordinates
* `max_horizon`: the maximum horizon of the STL Formula to learn
* `primitives` (optional): either `'MOTION_PLANNING'` or `'CLASSICAL'` (default set to `'CLASSICAL'`)
* `signal_gen` (optional): the signal generation method given an STL Formula. Either `'BOOLEAN'` or `'QUANTITATIVE'` or `'QUANTITATIVE_OPTIMIZE'` (default set to `'QUANTITATIVE_OPTIMIZE'`)
* `U` (optional): a basic control policy standing for the max amplitude of the signal between 2 time stamps
* `epsilon` (optional): basic control policy parameter (default set to `0.05`)
* `alpha` (optional): a convergence factor which we consider the distance between the hypothesis specification and target specification good enough to terminate the algorithm (default set to `0.01`)
* `beta` (optional): probability of triggering either a membership query or an equivalence query (default set to `0.5`)
* `gamma` (optional): number of iterations without improvement (decrease of distance between `phi_hypothesis` and `phi_target`) after triggering reset of the decision tree (default set to `50`)
* `MAX_IT` (optional): maximum number of iterations (default set to `100`)
* `phi_hypothesis` (optional): the hypothesis specification (default set to `True`)
* `plot_activated` (optional): show plot at each iteration (default set to `False`)

`active_learn.dtlearn.simple_boolean()` returns the Conjunctive Normal Form of the learnt STL Formula.