import pandas as pd
import GPy
from jupyterthemes import jtplot
import numpy as np
import pylab as pb
import matplotlib.pyplot as plt
jtplot.style(theme='default')


# funcion para plotear multi-task GP (Ricardo Andrade-Pacheco)
def plot_2outputs(modelo, xlim):
    fig = pb.figure(figsize=(12, 8))

    # Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    modelo.plot(plot_limits=xlim, fixed_inputs=[(1, 0)], which_data_rows=slice(0, 100), ax=ax1)

    # Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    modelo.plot(plot_limits=xlim, fixed_inputs=[(1, 1)], which_data_rows=slice(100, 200), ax=ax2)


# Datos finales para las predicciones
folder = '/Users/juanpablodonosomerlet/Desktop/estudio-acidos/data/'
xls_final = pd.ExcelFile(folder + '000 Datos Finales para analisis.xlsx')
df_datos_finales = pd.read_excel(xls_final, 'Datos', header=0, index_col=1)

# datos de produccion de fino por tonelada
t_cu_fino = df_datos_finales.index.values
t_cu_fino = np.reshape(t_cu_fino, newshape=(t_cu_fino.shape[0], 1))
t_cu_fino = t_cu_fino[:-5, 0]
t_cu_fino = np.reshape(t_cu_fino, newshape=(t_cu_fino.shape[0], 1))
prod_fino_final = df_datos_finales['Produccion Catodo\nton'].as_matrix()
prod_fino_final = np.reshape(prod_fino_final, newshape=(prod_fino_final.shape[0], 1))
prod_fino_final = prod_fino_final[:-5, 0]
prod_fino_final = np.reshape(prod_fino_final, newshape=(prod_fino_final.shape[0], 1))


# datos de consumo de acido
t_ca = df_datos_finales.loc[df_datos_finales.index.values <= 2017].index.values
t_ca = np.reshape(t_ca, newshape=(t_ca.shape[0], 1))
consumo_ac_final = df_datos_finales['Consumo Acido\nton'].loc[df_datos_finales.index.values <= 2017].as_matrix()
consumo_ac_final = np.reshape(consumo_ac_final, newshape=(consumo_ac_final.shape[0], 1))

# datos para la produccion de acido
t_pa = df_datos_finales.loc[df_datos_finales.index.values <= 2017].index.values
t_pa = np.reshape(t_pa, newshape=(t_pa.shape[0], 1))
prod_ac_final = df_datos_finales['Produccion de Acido'].loc[df_datos_finales.index.values <= 2017].as_matrix()
prod_ac_final = np.reshape(prod_ac_final, newshape=(prod_ac_final.shape[0], 1))

# datos para la ley de cobre
t_leyes = df_datos_finales.loc[(df_datos_finales.index.values >= 2003) &
                               (df_datos_finales.index.values <= 2016)].index.values
t_leyes = np.reshape(t_leyes, newshape=(t_leyes.shape[0], 1))
leyes_final = df_datos_finales['LEYES\nPromedios'].loc[(df_datos_finales.index.values >= 2003) &
                                                       (df_datos_finales.index.values <= 2016)].as_matrix()
leyes_final = np.reshape(leyes_final, newshape=(leyes_final.shape[0], 1))


if __name__ == '__main__':

    #

    K1 = GPy.kern.Bias(1)
    K2 = GPy.kern.Linear(1)
    K3 = GPy.kern.Matern32(1)
    lcm = GPy.util.multioutput.LCM(input_dim=1, num_outputs=2, kernels_list=[K1, K2, K3])

    m1 = GPy.models.GPCoregionalizedRegression([t_ca, t_cu_fino], [consumo_ac_final, prod_fino_final], kernel=lcm)

    m1['.*ICM.*var'].unconstrain()
    m1['.*ICM0.*var'].constrain_fixed(1.)
    m1['.*ICM0.*W'].constrain_fixed(0)
    m1['.*ICM1.*var'].constrain_fixed(1.)
    m1['.*ICM1.*W'].constrain_fixed(0)
    m1.optimize()
    plot_2outputs(m1, (1995, 2035))

    m2 = GPy.models.GPCoregionalizedRegression([t_leyes, t_cu_fino], [leyes_final, prod_fino_final], kernel=lcm)
    # m2['.*ICM.*var'].unconstrain()
    # m2['.*ICM0.*var'].constrain_fixed(1.)
    # m2['.*ICM0.*W'].constrain_fixed(0)
    # m2['.*ICM1.*var'].constrain_fixed(1.)
    # m2['.*ICM1.*W'].constrain_fixed(0)
    m2.optimize()
    plot_2outputs(m2, (1995, 2035))
    plt.show()
