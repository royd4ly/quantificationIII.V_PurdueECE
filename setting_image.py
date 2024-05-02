import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import atomap.api as am
import tkinter as tk
from tkinter.filedialog import askopenfilenames
from skimage.restoration import estimate_sigma
from sklearn import decomposition #PCA
from scipy import fft, signal
from skimage.filters import butterworth
from skimage.restoration import denoise_nl_means # NL means
from skimage.restoration import richardson_lucy # RL deconvolution

# Module to open, normalize and filter the image

def open_file():
    root = tk.Tk()
    root.attributes('-topmost',True)
    root.iconify()   
    file_path = askopenfilenames(parent=root)
    root.destroy()
    return file_path

class imp:
    def __init__(self, s,  pixels_cropped, pixel_size_pm, inner_angle, outer_angle):
        self.image = s
        self.data = s.data
        self.pixels_cropped = pixels_cropped
        self.pixel_size_pm = pixel_size_pm
        self.inner_angle = inner_angle
        self.outer_angle = outer_angle
        
    def scale(self, det_image):
        pixels_cropped  = self.pixels_cropped
        # Axes Scaling
        s = self.image
        s.axes_manager[0].name = 'X'
        s.axes_manager[1].name = 'Y'
        s.axes_manager[0].scale = self.pixel_size_pm/1000
        s.axes_manager[1].scale = self.pixel_size_pm/1000
        s.axes_manager[0].units = 'nm'
        s.axes_manager[1].units = 'nm'
        s.metadata.General.title = ''
        
        if det_image is None:
            print(det_image)
            print("Detecting image not finded")
            self.image = s
            self.data = self.data[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
        else:    
            # Intensity scaling
            s_normalised = am.quant.detector_normalisation(self.image, det_image, inner_angle=self.inner_angle, outer_angle=self.outer_angle)
            self.image = s_normalised
            self.image.data = s_normalised.data[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
            self.data = s_normalised.data[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]

        #plt.savefig(path+'\\normalized_image.png',dpi=512,transparent=True,bbox_inches='tight')

    def plot(self):
        self.image.plot(colorbar=False)
        ax=plt.gca()
        norm = mpl.colors.Normalize(vmin=np.min(self.image.data), vmax=np.max(self.image.data))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.15)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greys_r'),ax=ax, pad=.05, fraction=.1,cax=cax)
        plt.tight_layout()
        
        
    def PCA(self, n_components):
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(self.image.data)
        pcaFaces = pca.transform(self.image.data)
        pca_imag = pca.inverse_transform(pcaFaces)
        sigma_est = np.mean(estimate_sigma(pca_imag))
        print(sigma_est)
        return pca_imag

    def NL(self, sigma_est):
        h=5
        patch_size=5
        patch_distance=6
        nlm_imag = denoise_nl_means(self.data, h=h*sigma_est, fast_mode=True,patch_size=patch_size,patch_distance=patch_distance)
        return nlm_imag

    def RL(self, probe_resolution, iters):
        pixel_size=self.pixel_size_pm/1000
        width_pix=round(probe_resolution/pixel_size)
        kernlen = 4*width_pix
        std=width_pix/2.348
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gaussian_probe= gkern2d
        #plt.figure(figsize=(4,4))
        #plt.imshow(gaussian_probe,cmap='gray')
        rl_imag = richardson_lucy(self.data, gaussian_probe, iters)
        return rl_imag

    def band_pass(self):
        image = self.data
        #image = image[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
        high_pass=np.mean(np.abs(fft.ifft2((fft.fft2(image)==fft.fft2(image)[0,0])*fft.fft2(image))))+butterworth(image,0.005,True,order=2)
        band_pass=butterworth(high_pass,0.08,False,order=4)
        return band_pass

def compare(original_image, filter_imag, text):
    pixels_cropped = original_image.pixels_cropped
    original_imag = original_image.image.data
    filter_imag = filter_imag.image.data
    sigma_est_original = np.mean(estimate_sigma(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
    print(f'estimated noise standard deviation from original image = {sigma_est_original}')
    
    sigma_est = np.mean(estimate_sigma(filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
    print(f'estimated noise standard deviation from {text} denoising = {sigma_est}')
    
    fig, ax = plt.subplots(figsize=(12,8), nrows=1, ncols=3)
    
    ax[0].set_title('Original Image')
    ax[0].imshow(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[0].axis('off')
    
    
    ax[1].set_title(text)
    ax[1].imshow(filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[1].axis('off')
    
    ax[2].set_title('Residuals')
    ax[2].imshow(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]-filter_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
    ax[2].axis('off')
    
    fig.tight_layout()
    plt.show(block=False)

import hyperspy.api as hs
import os
if __name__ == '__main__':

    global pixels_cropped, pixel_size_pm, inner_angle, outer_angle
    pixel_size_pm=6.652 #3.326 #6.652 # pm
    inner_angle=130
    outer_angle=200
    pixels_cropped=8

    file_paths= open_file()
    det_path= open_file()
    det_image=hs.signals.Signal2D(plt.imread(det_path[0]))

    global dumbell
    dumbell = False

    for file_path in file_paths:
    
  
        s=hs.load(file_path)
        s = s.isig[500:1500, 500:3500]
        
        path = os.path.splitext(file_path)[0]
        if not (os.path.exists(path)):
            os.mkdir(path)
    
    SL = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)