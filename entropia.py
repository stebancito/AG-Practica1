import numpy as np
import cv2

def aperturaImagen(ruta):
    ##Trae la imagen sin ninguna modificación
    imagen = cv2.imread(ruta, cv2.IMREAD_UNCHANGED) 
    
    ## Si la imagen es de tres dimensiones, es de color
    if len(imagen.shape) == 3:
        ## Convertmos de BGR a RGB
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    ## Normalizamos con valores de pixeles decimales de doble precisión
    ##[0.0,1.0]
    imagen_double = imagen.astype(np.float64) / 255.0
    
    return imagen_double

def poblacionInicial(tam_poblacion, limites):

    ## Desempaquetado de límites inferiores
    lim_inf_alpha, lim_inf_delta = limites[0]
    ## Desempaquetado de límites superiores
    lim_sup_alpha, lim_sup_delta = limites[1]
    
    ## Se crean dos listas con distribución uniforme
    alphas = np.random.uniform(lim_inf_alpha, lim_sup_alpha, tam_poblacion)
    deltas = np.random.uniform(lim_inf_delta, lim_sup_delta, tam_poblacion)
    
    ## Se unen verticalmente las dos listas
    poblacion = np.column_stack((alphas, deltas))
    
    return poblacion

def tranSigmoide(imagen, alpha, delta):

    imagen_tran = 1.0 / (1.0 + np.exp(-alpha * (imagen - delta)))
    
    return imagen_tran

def entropia(imagen_tran):
    ## Desnormalización
    img_255 = (imagen_tran * 255).astype(np.uint8)
    
    ##Se crean 256 bins del rango de 0 255
    ## Solo queremos cuántos píxeles hay en cada bin, ignoramos los bordes 
    hist, _ = np.histogram(img_255.flatten(), bins=256, range=[0, 256])
    
    # Pk es la probabilidad de ocurrencia del nivel de intensidad
    pk = hist / hist.sum()
    
    ## Filtramos las pk mayores a 0 para evitar el error log(0)
    pk_no_cero = pk[pk > 0]
    
    entropia = -np.sum(pk_no_cero * np.log2(pk_no_cero))
    
    return entropia

def evaluarPoblacion(imagen_original, poblacion):
    
    ##Lista llena de ceros
    aptitudes = np.zeros(len(poblacion))
    
    for i in range(len(poblacion)):
        ##Individuo actual
        alpha = poblacion[i, 0]
        delta = poblacion[i, 1]
        
        img_transformada = tranSigmoide(imagen_original, alpha, delta)

        aptitudes[i] = entropia(img_transformada)
        
    return aptitudes

def selecPadres(aptitudes, poblacion):

    Np = len(poblacion)
    ## Matriz de ceros del mismo tamaño que poblacion
    padres = np.zeros_like(poblacion)

    
    ## Permutaciones aleatorias para seleccionar competidores (Padres)
    perm1 = np.random.permutation(Np)
    perm2 = np.random.permutation(Np)

    for i in range(Np):

        ## Elegimos competidores
        competidor1 = perm1[i]
        competidor2 = perm2[i]

        if aptitudes[competidor1] > aptitudes[competidor2]:
            # Pasa competidor de la izquierda
            padres[i, :] = poblacion[competidor1, :] 
        else:
            # Pasa competidor de la derecha
            padres[i, :] = poblacion[competidor2, :] 
            
    return padres


def cruzamientoSBX(padres, p_cruzamiento, limites, Nc):
    Np, variables = np.shape(padres)

    hijos = np.copy(padres)

    ##Bucle para seleccionar parejas, avanza de 2 en 2
    for i in range(0, Np-1, 2):
        
        ##Sacamos un random y si es menor o igual a Pc se cruza
        if np.random.rand() <= p_cruzamiento:
            ##Se genera el valor random de u (0-1)
            u = np.random.rand() 
            
            for j in range(variables):
                
                ## Forzamos que p1 sea el menor y p2 el mayor para la matematica
                p1 = min(padres[i, j], padres[i+1, j])
                p2 = max(padres[i, j], padres[i+1, j])

                ## Evitar division por cero si los genes son idénticos
                if abs(p2 - p1) < 1e-10:
                    continue 

                beta = 1.0 + 2.0 / (p2 - p1) * min(p1 - limites[0][j], limites[1][j] - p2) 
                alpha = 2.0 - abs(beta)**(-(Nc + 1.0))
                
                if u <= 1.0 / alpha:
                    beta_c = (u * alpha)**(1.0 / (Nc + 1.0))
                else:
                    beta_c = (1.0 / (2.0 - u * alpha))**(1.0 / (Nc + 1.0))
                
                hijo1 = 0.5 * ((p1 + p2) - beta_c * abs(p2 - p1))
                hijo2 = 0.5 * ((p1 + p2) + beta_c * abs(p2 - p1))
            
                hijos[i, j] = np.clip(hijo1, limites[0][j], limites[1][j])
                hijos[i+1, j] = np.clip(hijo2, limites[0][j], limites[1][j])

    return hijos

def mutacionPolinomial(poblacion, limites, p_mutacion, nm):
    Np, variables = np.shape(poblacion)
    mutados = np.copy(poblacion)

    for i in range(Np):
        for j in range(variables):
            
            rand = np.random.rand()
            if rand <= p_mutacion:
                
                r = np.random.rand()
                
                delta = min(limites[1][j] - mutados[i, j], mutados[i, j] - limites[0][j]) / (limites[1][j] - limites[0][j])
                
                if r <= 0.5:
                    delta_q = ((2.0 * r) + (1.0 - 2.0 * r) * (1.0 - delta)**(nm + 1.0))**(1.0 / (nm + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - ((2.0 * (1.0 - r)) + 2.0 * (r - 0.5) * (1.0 - delta)**(nm + 1.0))**(1.0 / (nm + 1.0))

                ## Mutar el gen  
                mutados[i, j] += delta_q * (limites[1][j] - limites[0][j])
                
                ## Aseguramos que el nuevo gen no rompa el espacio de búsqueda
                mutados[i, j] = np.clip(mutados[i, j], limites[0][j], limites[1][j])

    return mutados
    

def sustitucionExEl(poblacion, hijos, entropia_hijos, entropia_poblacion):
    ## Genes del mejor individuo
    mejor_idividuo = poblacion[np.argmax(entropia_poblacion)]

    idx_peor_hijo = np.argmin(entropia_hijos)

    hijos[idx_peor_hijo] = mejor_idividuo
    return hijos

def algoritmoGenetico(limites, tam_poblacion, generaciones, p_cruzamiento, p_mutacion, nc, nm, ruta):
    
    ## Abrir la imagen
    imagen = aperturaImagen(ruta)

    ## Generar la población inicial
    poblacion = poblacionInicial(tam_poblacion, limites)

    entropia_poblacion = evaluarPoblacion(imagen, poblacion)

    for i in range(generaciones):

        ## Seleccion de padres por torneo determinista
        padres = selecPadres(entropia_poblacion, poblacion)

        ## Cruzamiento por SBX
        hijos = cruzamientoSBX(padres, p_cruzamiento,limites, nc)

        ## Mutación polinomial
        hijos_mutados = mutacionPolinomial(hijos, limites, p_mutacion, nm)

        ## Evaluar entropía de los hijos mutados
        entropia_hijos = evaluarPoblacion(imagen, hijos_mutados)

        ## Sustitución elitista
        poblacion = sustitucionExEl(poblacion, hijos_mutados, entropia_hijos, entropia_poblacion)

        ## Evaluar entropía de la nueva población
        entropia_poblacion = evaluarPoblacion(imagen, poblacion)

        mejor_individuo = poblacion[np.argmax(entropia_poblacion)]
        mejor_entropia = np.max(entropia_poblacion)

        print(f"Generación {i+1}: Mejor individuo = {mejor_individuo}, Entropía = {mejor_entropia:.4f}")

    return mejor_individuo, mejor_entropia

ruta = "White blood cell (basophil).jpg"

tam_poblacion = 50
generaciones = 40
p_cruzamiento = 0.9
p_mutacion = 0.2
nc = 9
nm = 50
limites = [[0.0, 0.0], [10.0, 1.0]]
# mejor_individuo, mejor_entropia = algoritmoGenetico(limites, tam_poblacion, generaciones, p_cruzamiento, p_mutacion, nc, nm, ruta)

# imagen = aperturaImagen(ruta)
# print(f"\nMejor solución encontrada: Alpha = {mejor_individuo[0]:.4f}, Delta = {mejor_individuo[1]:.4f}")

# imagen_final = tranSigmoide(imagen, mejor_individuo[0], mejor_individuo[1])

# imagen_final_8bit = (imagen_final * 255).astype(np.uint8)

# # Si la imagen es de 3 dimensiones, la regresamos a BGR para guardar
# if len(imagen_final_8bit.shape) == 3:
#     imagen_guardar = cv2.cvtColor(imagen_final_8bit, cv2.COLOR_RGB2BGR)
# else:
#     imagen_guardar = imagen_final_8bit

# cv2.imwrite("imagen_optima_.jpg", imagen_guardar)
# print("¡Imagen guardada con éxito!")

historial_soluciones = []

for i in range(10):
    print(f"   EJECUCIÓN {i + 1} DE 10")
    mejor_individuo, mejor_aptitud = algoritmoGenetico(limites, tam_poblacion, generaciones, p_cruzamiento, p_mutacion, nc, nm, ruta)
    historial_soluciones.append({
        "Ejecucion": i + 1,
        "Alpha": mejor_individuo[0],
        "Delta": mejor_individuo[1],
        "Aptitud": mejor_aptitud
    })
    print(f"Mejor solución encontrada: Alpha = {mejor_individuo[0]:.4f}, Delta = {mejor_individuo[1]:.4f}, Aptitud = {mejor_aptitud:.4f}")
    imagen = aperturaImagen(ruta)
    imagen_final = tranSigmoide(imagen, mejor_individuo[0], mejor_individuo[1])
    imagen_final_8bit = (imagen_final * 255).astype(np.uint8)
    
    if len(imagen_final_8bit.shape) == 3:
        imagen_guardar = cv2.cvtColor(imagen_final_8bit, cv2.COLOR_RGB2BGR)
    else:
        imagen_guardar = imagen_final_8bit

    nombre_archivo = f"imagen_optima_ejecucion_globulo_{i + 1}.jpg"
    cv2.imwrite(nombre_archivo, imagen_guardar)
    print(f"¡Imagen guardada con éxito como: {nombre_archivo}!")

print("RESUMEN DE LAS 10 EJECUCIONES PARA EL REPORTE")
for res in historial_soluciones:
    print(f"Ejecución {res['Ejecucion']}: Alpha={res['Alpha']:<7.4f} | Delta={res['Delta']:<7.4f} | Aptitud={res['Aptitud']:.4f}")
    