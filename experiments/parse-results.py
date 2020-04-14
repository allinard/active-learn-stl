import pickle
from sympy.logic import simplify_logic, bool_map

posf = open("pos.pickle",'rb')
negf = open("neg.pickle",'rb')
dicf = open("dic.pickle",'rb')
dic2f = open("dic2.pickle",'rb')
timf = open("tim.pickle",'rb')
pos = pickle.load(posf)
neg = pickle.load(negf)
dic = pickle.load(dicf)
dic2 = pickle.load(dic2f)
tim = pickle.load(timf)
posf.close()
negf.close()
dicf.close()
dic2f.close()
timf.close()


f = open("results.csv", "w")


distance = float("inf")
for iteration in dic:

    if dic[iteration][0] < distance:
        distance = dic[iteration][0]

        print("processing iteration ",iteration)

        f.write(str(iteration))
        f.write(';')
        
        f.write(str(dic[iteration][0]))
        f.write(';')
        
        f.write(dic[iteration][1])
        f.write(';')    
        
        f.write(dic[iteration][2])
        f.write(';')
        
        try:
            f.write(str(simplify_logic(expr=dic[iteration][2],form='cnf')))
            f.write(';')
        except AttributeError:
            f.write('True;')
        
        f.write(str((tim/1000)*iteration))
        f.write('\n')


f.write('\n\n')

for entry in dic2:
    f.write(entry)
    f.write(';')
    f.write(dic2[entry])
    f.write('\n')

f.close()