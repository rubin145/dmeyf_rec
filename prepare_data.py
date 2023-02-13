import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#para FE
def make_premiums(x):
    features = ['cprestamos_personales','cprestamos_prendarios','cprestamos_hipotecarios','cplazo_fijo','cinversion1','cinversion2','cseguro_vida','cseguro_auto','cseguro_vivienda','cseguro_accidentes_personales','ccaja_seguridad']
    values = [x[f] for f in features]
    return sum(values)

def make_estables_tr(x):
    features = ['ctarjeta_debito_transacciones','ctarjeta_visa_transacciones','ctarjeta_master_transacciones','cpayroll_trx','ctarjeta_master_debitos_automaticos','ctarjeta_visa_debitos_automaticos']
    values = [x[f] for f in features]
    return sum(values)

df = pd.read_csv('data_origen/rubinstein_generacion.txt.gz',sep='\t')

df['clase01'] = df.clase.apply(lambda x: 1 if x=='SI' else 0)
del df['clase']
print('clase pasada a 0,1')

df['cpremiums'] = df.apply(make_premiums,axis=1)
df['premiums_bool'] = df.cpremiums.apply(lambda x : x > 0)

df['estables_tr'] = df.apply(make_estables_tr,axis=1)
df['estables_tr_bool'] = df.estables_tr.apply(lambda x : x > 0)
print('variables nuevas creadas')

labels = np.array(df['clase01'])
features = np.array(df.drop(columns=['clase01']))
feature_names = list(df.drop(columns=['clase01']).columns)
del df
print('chau pandas hola numpy')

np.save('data_proceso/X_dev.npy', features)
np.save('data_proceso/y_dev.npy', labels)
np.save('data_proceso/feature_names.npy', feature_names)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
print('hecho el split')

np.save('data_proceso/X_train.npy', X_train)
np.save('data_proceso/X_test.npy', X_test)
np.save('data_proceso/y_train.npy', y_train)
np.save('data_proceso/y_test.npy', y_test)

np.savetxt('data_proceso/test_ids.csv', X_test[:,0], fmt='%d')

df_aplic = pd.read_csv('data_origen/rubinstein_aplicacion.txt.gz',sep='\t')

df_aplic['cpremiums'] = df_aplic.apply(make_premiums,axis=1)
df_aplic['premiums_bool'] = df_aplic.cpremiums.apply(lambda x : x > 0)

df_aplic['estables_tr'] = df_aplic.apply(make_estables_tr,axis=1)
df_aplic['estables_tr_bool'] = df_aplic.estables_tr.apply(lambda x : x > 0)

features_aplic = np.array(df_aplic)

np.save('data_proceso/X_aplic.npy', features_aplic)

def test_equal(object_1, file, allow_pickle=True):
    print(f'testeando {get_var_name(object_1)} con {file}')
    object_2 = np.load(os.path.join(dir,file),allow_pickle=allow_pickle)

    print('cuántos NaN')
    nans_1 = np.isnan(object_1.astype(float)).sum()
    nans_2 = np.isnan(object_2.astype(float)).sum()
    print(nans_1)
    print(nans_2)

    print('dtype')
    print(object_1.dtype)
    print(object_2.dtype)

    print('shape')
    print(object_1.shape)
    print(object_2.shape)

    print('datapoints')
    if len(object_1.shape) == 2:
        dp_1 = object_1.shape[0] * object_1.shape[1]
        dp_2 = object_2.shape[0] * object_2.shape[1]
    elif len(object_1.shape) == 1:
        dp_1 = object_1.shape[0]
        dp_2 = object_2.shape[0]
    print(dp_1)
    print(dp_2)

    print('cuántos iguales')
    if len(object_1.shape) == 2:
        iguales = sum(sum(object_1 == object_2))
    elif len(object_1.shape) == 1:
        iguales = sum(object_1 == object_2)
    print(iguales)

    print('coincide?')
    print(iguales == dp_1 - nans_1)
    print(iguales == dp_2 - nans_2)

def test_equal_list(list_1, file, allow_pickle=True, dir='data_proceso'):
    print(f'testeando {get_var_name(list_1)} con {file}')
    list_2 = np.load(os.path.join(dir,file),allow_pickle=allow_pickle)
    print('coincide?')
    print(sum(list_1==list_2) == len(list_1) == len(list_2))

def get_var_name(var):
    for name in globals():
        if globals()[name] is var:
            return name
    for name in locals():
        if locals()[name] is var:
            return name
    return "Variable not found"

#test_equal(X_train, 'X_train.npy')
#test_equal(X_test, 'X_test.npy')
#test_equal(y_train, 'y_train.npy')
#test_equal(y_test, 'y_test.npy')
#test_equal_list(feature_names, 'feature_names.npy')
#test_equal_list(y_train, 'y_train.npy')
#test_equal_list(y_test, 'y_test.npy')
#test_equal(y_train, 'y_train.npy', allow_pickle=False)
#test_equal(y_test, 'y_test.npy', allow_pickle=False)
#test_equal_list(feature_names, 'feature_names.npy', allow_pickle=False)
#test_equal(X_train, 'X_train.npy', allow_pickle=False) #falla
#test_equal(X_test, 'X_test.npy', allow_pickle=False) #falla

