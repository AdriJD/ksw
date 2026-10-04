from Afunctionals import products_3j_array

S = 1
n = 1
L_list = range(2, 20)
deltaL_list = [-1, 1]
Jindex = (0, 0, 0)

products = products_3j_array(S, n, L_list, deltaL_list, Jindex)
print(products)