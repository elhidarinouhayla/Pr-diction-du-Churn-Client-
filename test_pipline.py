from pipline import charge_donnees, pretraitement_variable, split_features


# Vérifier que le nombre de lignes est le même

data = charge_donnees()
data_clean = pretraitement_variable(data)
x, y = split_features(data_clean)

if len(x) == len(y):
    print("les demensions de x et y sont coherentes ")
else:
    print("X et y n'ont pas le même nombre de lignes")