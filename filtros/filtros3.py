import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy
from functools import partial
from scipy.ndimage import convolve
from scipy import ndimage



class ImageProcessorApp:
    image = None
    file_path = None
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")
        self.root.geometry("1300x700")
        self.etiqueta = tk.Label(root, text="Procesamiento digital de imágenes")
        self.etiqueta.grid(row=0, column=3, columnspan=4)

        self.red = tk.Label(root, text="Red:")
        self.red.grid(row=9, column=1)

        self.green = tk.Label(root, text="Green:")
        self.green.grid(row=9, column=2)

        self.blue = tk.Label(root, text="Blue:")
        self.blue.grid(row=9, column=3)
        # Frame izquierdo para etiquetas y botones

        self.load_button = tk.Button(root, text="   Abrir imagen      ", command=self.load_image)
        self.load_button.grid(row=3, column=10, rowspan=2)

        self.grayscale_button = tk.Button(root, text=" Escala de grises   ", command=self.grayscale_image)
        self.grayscale_button.grid(row=4, column=10, rowspan=2)

        self.binarize_button = tk.Button(root, text="     Binarizar      ", command=self.binarizador)
        self.binarize_button.grid(row=5, column=10, rowspan=2)

        self.dynamic_binarize_button = tk.Button(root, text="Binarizador dinámico", command=self.binarizadorDina)
        self.dynamic_binarize_button.grid(row=6, column=10, rowspan=2)

        self.negative_filter_button = tk.Button(root, text="  cambiar a negativo  ", command=self.filtroNegativo)
        self.negative_filter_button.grid(row=7, column=10, rowspan=2)

        self.smooth_filter_button = tk.Button(root, text="  difuminar 9x9  ", command=self.smooth_image)
        self.smooth_filter_button.grid(row=8, column=10, rowspan=2)

        self.nolineal_filter_button = tk.Button(root, text="  difuminar ponderados  ", command=self.nolineal_image)
        self.nolineal_filter_button.grid(row=9, column=10, rowspan=2)

        self.sobel_filter_button = tk.Button(root, text="  sobel  ", command=self.sobel_edge_detection)
        self.sobel_filter_button.grid(row=11, column=12, rowspan=2)

        self.prewitt_filter_button = tk.Button(root, text="  prewitt  ", command=self.prewitt_edge_detection)
        self.prewitt_filter_button.grid(row=11, column=11, rowspan=2)

        self.canny_filter_button = tk.Button(root, text="  canny  ", command=self.canny_edge_detection)
        self.canny_filter_button.grid(row=11, column=13, rowspan=2)

        self.roberts_filter_button = tk.Button(root, text="  roberts  ", command=self.roberts_edge_detection)
        self.roberts_filter_button.grid(row=11, column=14, rowspan=2)
        # canvas1

        frame = tk.Frame(root, borderwidth=1, relief="solid")
        frame.grid(row=3, column=0, rowspan=6, columnspan=4, padx=(30, 20), pady=(30, 30))

        y_scrollbar = tk.Scrollbar(frame, orient="vertical")
        y_scrollbar.pack(side="right", fill="y")

        x_scrollbar = tk.Scrollbar(frame, orient="horizontal")
        x_scrollbar.pack(side="bottom", fill="x")

        self.canvas1 = tk.Canvas(frame, width=512, height=512, yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        self.canvas1.pack(expand=True, fill="both")
        self.canvas1.bind("<Motion>", self.on_mouse_move)
        y_scrollbar.config(command=self.canvas1.yview)
        x_scrollbar.config(command=self.canvas1.xview)
        self.canvas1.config(scrollregion=self.canvas1.bbox("all"))


        # canvas2
        frame2 = tk.Frame(root, borderwidth=1, relief="solid")
        frame2.grid(row=3, column=6, rowspan=6, columnspan=4, padx=(0, 20), pady=(30, 30))

        y_scrollbar = tk.Scrollbar(frame2, orient="vertical")
        y_scrollbar.pack(side="right", fill="y")

        x_scrollbar = tk.Scrollbar(frame2, orient="horizontal")
        x_scrollbar.pack(side="bottom", fill="x")

        self.canvas2 = tk.Canvas(frame2, width=512, height=512, yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        self.canvas2.pack(expand=True, fill="both")
        self.canvas2.bind("<Motion>", self.on_mouse_move)
        self.canvas2.bind("<Button-1>", partial(self.on_frame_click, dato=1, frame1=frame, frame2=frame2))
        self.canvas1.bind("<Button-1>", partial(self.on_frame_click, dato=0, frame1=frame, frame2=frame2))

        y_scrollbar.config(command=self.canvas2.yview)
        x_scrollbar.config(command=self.canvas2.xview)
        self.canvas2.config(scrollregion=self.canvas2.bbox("all"))

        self.image1 = None
        self.image2 = None
        self.canvas = None
#-------cargar y mostrar imagenes---------------
    def load_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Archivos PNG", "*.png"), ("Archivos JPG", "*.jpg")])
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                if self.canvas == self.canvas1:
                    self.image1 = image
                elif self.canvas == self.canvas2:
                    self.image2 = image
                self.display_image()  # Asegúrate de llamar a display_image() después de cargar la imagen

    def display_image(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

            if image is not None:
                img_rgb = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk
                self.canvas.config(scrollregion=self.canvas.bbox("all"))

#------------------------------------------------------------------------
    def on_mouse_move(self, event):
        if self.image is None:
            return
        x_canvas = event.x
        y_canvas = event.y
        if y_canvas < len(self.image) and x_canvas < len(self.image[0]):
            self.red.config(text="Red:" + str(self.image[y_canvas][x_canvas][2]))
            self.green.config(text="Green:" + str(self.image[y_canvas][x_canvas][1]))
            self.blue.config(text="Blue:" + str(self.image[y_canvas][x_canvas][0]))
    def on_frame_click(self, event, dato, frame1, frame2):
        if dato == 0:
            frame1.config(borderwidth=0, relief="solid", highlightbackground="green", highlightthickness=3)
            frame2.config(borderwidth=0, relief="solid", highlightbackground="black", highlightthickness=1)
            self.canvas = self.canvas1
            self.display_image()
        elif dato == 1:
            frame1.config(borderwidth=0, relief="solid", highlightbackground="black", highlightthickness=1)
            frame2.config(borderwidth=0, relief="solid", highlightbackground="green", highlightthickness=3)
            self.canvas = self.canvas2
            self.display_image()

#----------------filtros------------------------------------------------
    def grayscale_image(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

            if image is not None:
                filas, columnas, canales = image.shape
                for f in range(filas):
                    for c in range(columnas):
                        promedio = (int(image[f, c, 0]) + int(image[f, c, 1]) + int(image[f, c, 2])) // 3
                        image[f, c, 0] = numpy.uint8(promedio)
                        image[f, c, 1] = numpy.uint8(promedio)
                        image[f, c, 2] = numpy.uint8(promedio)

                self.display_image()
                print("Grayscale filter applied successfully.")
        


    def binarizador(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

        if image is not None:
            threshold = 128  # Umbral fijo
            filas, columnas, canales = image.shape
            for f in range(filas):
                for c in range(columnas):
                    promedio = (int(image[f, c, 0]) + int(image[f, c, 1]) + int(image[f, c, 2])) // 3
                    if promedio >= threshold:
                        image[f, c] = [255, 255, 255]
                    else:
                        image[f, c] = [0, 0, 0]
            self.display_image()

    def binarizadorDina(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

        if self.image is not None:
            filas, columnas, canales = image.shape
            promedioT = numpy.mean(image)
            threshold = int(promedioT)
            # Binarizar la imagen utilizando el umbral calculado
            for f in range(filas):
                for c in range(columnas):
                    promedio = (int(image[f, c, 0]) + int(image[f, c, 1]) + int(image[f, c, 2])) // 3
                    if promedio >= threshold:
                        image[f, c] = [255, 255, 255]
                    else:
                        image[f, c] = [0, 0, 0]

            self.display_image()

    def filtroNegativo(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

        if image is not None:
            filas, columnas, canales = image.shape
            for f in range(filas):
                for c in range(columnas):
                    for canal in range(canales):
                        image[f, c, canal] = 255 - image[f, c, canal]
            self.display_image()
    def smooth_image(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

        filas, columnas, canales = image.shape
        # Creamos una copia de la imagen original
        new_image = numpy.ndarray(shape=image.shape, dtype=numpy.uint8)
        for f in range(filas):
            for c in range(columnas):
                # Para una máscara de 9x9, ajustamos los límites de la región para incluir 4 píxeles adicionales a cada lado
                f1, f2 = f - 4, f + 5
                c1, c2 = c - 4, c + 5
                # Ajustamos los límites de la región si están fuera de la imagen
                if f1 < 0: f1 = 0
                if c1 < 0: c1 = 0
                if f2 > filas: f2 = filas
                if c2 > columnas: c2 = columnas
                # Tomamos la región de 9x9 alrededor del píxel actual
                region = image[f1:f2, c1:c2]
                # Calculamos el promedio de los valores de los píxeles en la región
                promedio = numpy.mean(region, axis=(0, 1))
                # Asignamos el promedio como nuevo valor para el píxel actual
                new_image[f, c] = numpy.uint8(promedio)

        image = new_image
        self.display_image()
#------------filtros de reconocimiento de imagenes--------------------------------
    def nolineal_image(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2
            if image is not None:  # Verificar si la imagen no es None después de asignarla
                filas, columnas, canales = image.shape
                mask = [[-1, 1, 1],
                        [-1, -2, 1],
                        [-1, 1, 1]]
                mask = numpy.ndarray(shape=(3, 3), dtype=int, buffer=numpy.array(mask, dtype=int))
                mag_image = numpy.ndarray(shape=(filas, columnas), dtype=int)
                for f in range(1, filas - 1):
                    for c in range(1, columnas - 1):
                        f1, f2 = f - 1, f + 1
                        c1, c2 = c - 1, c + 1
                        region = image[f1:f2 + 1, c1:c2 + 1, 0]
                        multi = mask * region
                        suma = multi.sum()
                        mag_image[f, c] = suma
                _max = mag_image.max()
                _min = mag_image.min()
                # Verificamos si _max - _min es 0 para evitar una posible división por cero
                if _max - _min != 0:
                    # Regla de 3 para mapear valores de vuelta al rango [0, 255]
                    mag_image = (mag_image - _min) * 255 / (_max - _min)
                else:
                    # Si _max y _min son iguales, no podemos realizar la operación de escalado,
                    # así que establecemos todos los valores de mag_image a 0
                    mag_image[:, :] = 0
                # Creamos una nueva imagen utilizando solo el canal de intensidad (escala de grises)
                new_image = numpy.uint8(mag_image)
                # Actualizamos la imagen en el lienzo correspondiente
                if self.canvas == self.canvas1:
                    self.image1 = cv2.merge((new_image, new_image, new_image))
                elif self.canvas == self.canvas2:
                    self.image2 = cv2.merge((new_image, new_image, new_image))
                self.display_image()



    
    def prewitt_edge_detection(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2
                
            if image is not None:
                # Definir los kernels para el filtro de detección de bordes Prewitt
                kernel_x = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                kernel_y = numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

                # Aplicar los kernels de Prewitt en ambas direcciones (horizontal y vertical)
                gradient_x = self.apply_sobel_kernel(image, kernel_x)
                gradient_y = self.apply_sobel_kernel(image, kernel_y)

                # Calcular el gradiente total
                gradient_magnitude = numpy.sqrt(gradient_x**2 + gradient_y**2)

                # Escalar el gradiente total al rango [0, 255]
                gradient_magnitude *= 255.0 / gradient_magnitude.max()

                # Actualizar la imagen en el lienzo correspondiente
                if self.canvas == self.canvas1:
                    self.image1 = numpy.stack((gradient_magnitude,) * 3, axis=-1).astype(numpy.uint8)
                elif self.canvas == self.canvas2:
                    self.image2 = numpy.stack((gradient_magnitude,) * 3, axis=-1).astype(numpy.uint8)

                self.display_image()


 

    def roberts_edge_detection(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2
                
            if image is not None:
                # Definir los kernels para el filtro de detección de bordes Roberts
                kernel_x = numpy.array([[1, 0], [0, -1]])
                kernel_y = numpy.array([[0, 1], [-1, 0]])

                # Aplicar los kernels de Roberts en ambas direcciones (horizontal y vertical)
                gradient_x = self.apply_roberts_kernel(image, kernel_x)
                gradient_y = self.apply_roberts_kernel(image, kernel_y)

                # Calcular el gradiente total
                gradient_magnitude = numpy.sqrt(gradient_x**2 + gradient_y**2)

                # Escalar el gradiente total al rango [0, 255]
                gradient_magnitude *= 255.0 / gradient_magnitude.max()

                # Actualizar la imagen en el lienzo correspondiente
                if self.canvas == self.canvas1:
                    self.image1 = numpy.stack((gradient_magnitude,) * 3, axis=-1).astype(numpy.uint8)
                elif self.canvas == self.canvas2:
                    self.image2 = numpy.stack((gradient_magnitude,) * 3, axis=-1).astype(numpy.uint8)

                self.display_image()


    def sobel_edge_detection(self):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

            if image is not None:
                # Convertir la imagen a escala de grises
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Aplicar el filtro de Sobel para calcular gradientes en las direcciones x e y
                sobel_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                # Aplicar los filtros de Sobel en ambas direcciones (horizontal y vertical)
                gradient_x = self.apply_sobel_kernel(image, sobel_x)
                gradient_y = self.apply_sobel_kernel(image, sobel_y)

                # Calcular el gradiente total
                gradient_magnitude = numpy.sqrt(gradient_x ** 2 + gradient_y ** 2)

                # Escalar el gradiente total al rango [0, 255]
                gradient_magnitude *= 255.0 / gradient_magnitude.max()

                # Convertir la imagen de bordes en una imagen RGB
                edges_rgb = numpy.stack((gradient_magnitude,) * 3, axis=-1).astype(numpy.uint8)

                # Actualizar la imagen en el lienzo correspondiente
                if self.canvas == self.canvas1:
                    self.image1 = edges_rgb
                elif self.canvas == self.canvas2:
                    self.image2 = edges_rgb

                self.display_image()
    def canny_edge_detection(self, kernel_size=5, sigma=0.4, low_threshold=50, high_threshold=150):
        if self.canvas is not None:
            if self.canvas == self.canvas1:
                image = self.image1
            elif self.canvas == self.canvas2:
                image = self.image2

            if image is not None:
                # Paso 1: Suavizado de la imagen
                smoothed_image = self.gaussian_blur(image, kernel_size, sigma)
                
                # Paso 2: Detección de gradientes
                gradient_magnitude, gradient_direction = self.sobel_filters(smoothed_image, kernel_size, sigma)
                
                # Paso 3: Supresión de no máximos
                suppressed_gradient = self.non_max_suppression(gradient_magnitude, gradient_direction)
                
                # Paso 4: Umbralización por histéresis
                edge_image = self.hysteresis(suppressed_gradient, low_threshold, high_threshold)
                
                if self.canvas == self.canvas1:
                    self.image1 = edge_image
                elif self.canvas == self.canvas2:
                    self.image2 = edge_image

                self.display_image()
    def gaussian_kernel(self,size, sigma):

        kernel_1D = numpy.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = numpy.exp(-0.5 * (kernel_1D[i] / sigma) ** 2)
        kernel_2D = numpy.outer(kernel_1D.T, kernel_1D.T)
        kernel_2D *= 1.0 / kernel_2D.max()
        return kernel_2D

    def gaussian_blur(self,image, kernel_size=5, sigma=1.4):
        kernel = self.gaussian_kernel(kernel_size, sigma)
        blurred_image = numpy.zeros_like(image, dtype=float)
        for channel in range(image.shape[2]):
            blurred_image[:, :, channel] = convolve(image[:, :, channel], kernel)
        return blurred_image.astype(numpy.uint8)

    def sobel_filters(self, img, kernel_size, sigma):
        # Definir los kernels de Sobel
        Kx = numpy.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
        Ky = numpy.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]])

        # Aplicar los filtros de Sobel
        Ix = self.apply_sobel_kernel(img, Kx)
        Iy = self.apply_sobel_kernel(img, Ky)

        
        G = numpy.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = numpy.arctan2(Iy, Ix)
        
        return (G, theta)

    def non_max_suppression(self,img, D):
        M, N = img.shape  
        Z = numpy.zeros((M,N), dtype=numpy.int32)
        angle = D * 180. / numpy.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255
                # Angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

        return Z

    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        
        M, N = img.shape
        res = numpy.zeros((M,N), dtype=numpy.int32)
        
        weak = numpy.int32(25)
        strong = numpy.int32(255)
        
        strong_i, strong_j = numpy.where(img >= highThreshold)
        zeros_i, zeros_j = numpy.where(img < lowThreshold)
        
        weak_i, weak_j = numpy.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return (res, weak, strong)
    
    def hysteresis(self,img, weak, strong=255):
        M, N = img.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img
#------------------------------------------------------------------------
#--------------funciones que aplican las mascaras a la imgaen----------------------
    def apply_sobel_kernel(self, image, kernel):
        # Aplicar el kernel a la imagen
        output = numpy.zeros_like(image[:, :, 0])
        rows, cols = image.shape[:2]
        ksize = kernel.shape[0]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image[i - 1:i + 2, j - 1:j + 2, 0]
                gradient = numpy.sum(region * kernel)
                output[i, j] = gradient

        # Normalizar los valores resultantes al rango [0, 255]
        output = numpy.abs(output)
        max_value = numpy.max(output)
        if max_value != 0:
            output = (output / max_value) * 255.0
        else:
            output = numpy.zeros_like(output)

        # Convertir el resultado a tipo uint8
        output = output.astype(numpy.uint8)

        return output

    def apply_roberts_kernel(self, image, kernel):
        output = numpy.zeros_like(image[:, :, 0])
        rows, cols = image.shape[:2]
        ksize = kernel.shape[0]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image[i - 1:i + 1, j - 1:j + 1, 0]  # Ajustar tamaño de la región
                gradient = numpy.sum(region * kernel)
                output[i, j] = gradient

        output = numpy.abs(output)
        max_value = numpy.max(output)
        if max_value != 0:
            output = (output / max_value) * 255.0
        else:
            output = numpy.zeros_like(output)

        output = output.astype(numpy.uint8)

        return output



root = tk.Tk()
app = ImageProcessorApp(root)
root.mainloop()
