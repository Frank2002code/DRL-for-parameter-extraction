import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def getIV (file):
    
    df = pd.read_csv(file)
    Id = df.id.values
    Vg = df.vg.values
    b = file.split("_")  
    b = b[2].split(".csv") 
    display(df)
    print('Vds = ',float(b[0]))  
    
    return Id, Vg, float(b[0])

# RMS
def rms(id_model, id_data):

    size = np.size(id_model)
    a = (id_data**2).sum()
    target = (a/size)**0.5
    b = ((abs(id_model-id_data)/target)**2).sum()
    RMS = ((b/size)**0.5)*100
    
    return RMS

def plot_IV_curve(vg, id, vgs, id_fit, title):
#     fig = plt.figure(figsize=(20, 8))

    fig = plt.figure()
    plt.title(title+"_log", fontsize=20)
    plt.xlabel("Gate Voltage (V)", fontsize=13)
    line1, = plt.plot(vg, id, color = 'blue', label = 'Experiments',markevery=3, linestyle=' ', marker='o')             
    line2, = plt.plot(vgs, id_fit, color = 'red', linewidth = 3, label = 'Modeling')
    plt.legend(handles = [line1, line2], loc='upper right')
    plt.show() 
        
    fig = plt.figure()
    plt.title(title+"_log", fontsize=20)
    plt.xlabel("Gate Voltage (V)", fontsize=13)
    line1, = plt.plot(vg, id, color = 'blue', label = 'Experiments',markevery=3, linestyle=' ', marker='o')             
    line2, = plt.plot(vgs, id_fit, color = 'red', linewidth = 3, label = 'Modeling')
    plt.legend(handles = [line1, line2], loc='upper right')
    plt.yscale('log')                 #####log scale
    plt.show()    
    
    return

def FIT(vgs, Id, Fit_paras, Init_val, param_bounds, sweep_bias, temp, modelcard):   
#     try:
#         import bsimcmg
#     except ModuleNotFoundError:
#         print('Warning: bsimcmg module from verilog-ae not found')
    import verilogae
    model =verilogae.load('./eehemt/eehemt114_2.va')



    global Fit_para
    Fit_para=Fit_paras      #### linear_func透過bsim 產生了計算的curve與 vgs去做fitting

    def func(vgs, *Fit_vals):
        
        for idx, y_col in enumerate(Fit_vals):    #設定更新的參數，paras vs iteration process 可print 出來 print(Fit_para[idx],y_col)
            modelcard[Fit_para[idx]]=y_col
            
        return model.functions['I_ds'].eval(temperature = temp, voltages = sweep_bias,**modelcard)    # initial value
    ## p0是linear_func(bsim函數模型)初始給的參數，vgs, Id 分別為ground truth 的x y
    ## maxfev (int) – Maximum allowed number of function evaluations. If both maxiter and maxfev are set, minimization will stop at the first reached.


    popt,pcov = curve_fit(func, vgs, Id, Init_val, bounds=param_bounds ,method='trf', maxfev = 100000)
#     popt,pcov = curve_fit(linear_function, linear_vg_300K, linear_id_300K, linear_p0, bounds=linear_param_bounds ,method='trf', maxfev = 100000)

    return popt

def BSIM_fit(Id, Vg, Vds, Fit_paras, Init_val, param_bounds,  state, temp, modelcard):
    import verilogae
    model =verilogae.load('./eehemt/eehemt114_2.va')

    Vgs  = Vg                              #!!!!!!!!!!!!!!!!!!!是不是就等於 Vg ?
    Vds = np.full_like(Vgs, Vds)          #!!!!!!!!!!!!!!!!!!!設為 vds ?
    Ves  = np.full_like(Vgs, 0.0)          #!!!!!!!!!!!!!!!!!!!設為 0 ?
    t    = np.full_like(Vgs, 10)
    edi    = np.full_like(Vgs, 10)

    sweep_bias = {
            'br_gisi': Vgs,
            'br_disi': Vds,
            'br_esi': Ves,
            'br_t': t,
            'br_edi': edi}    
    
    Id = Id.tolist()
    Vg = Vg.tolist()

    
    print("============================")
    print("=== Initial parameters ===")
    for idx, para in enumerate(Fit_paras):
        print(para, " = ", Init_val[idx])
    print("============================")

    updated_paras = FIT(Vgs, Id, Fit_paras, Init_val, param_bounds, sweep_bias, temp, modelcard)
    
    print("=== Optimized parameters ===")
    for idx, para in enumerate(Fit_paras):
        print(para, " = ", updated_paras[idx])
    print("============================")
    
    Id_fit = model.functions['I_ds'].eval(temperature = temp, voltages = sweep_bias, **modelcard)

    plot_IV_curve(Vg, Id, Vgs, Id_fit, state)
    
    return updated_paras, Id_fit
