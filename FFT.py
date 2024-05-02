import hyperspy.api as hs
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import atomap.api as am
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os
from skimage.restoration import estimate_sigma
from sklearn import decomposition
from scipy import fft
from skimage.filters import butterworth
from matplotlib.widgets import Slider
from matplotlib.widgets import Button, Slider


hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
plt.rcParams['figure.figsize'] = (8,8)

def plot_FFT(image):
    spectrum=fft.fftshift(fft.fft2(image))
    spectrum_modul = np.log(np.abs(spectrum))
    spectrum_phase = np.angle(spectrum)
    fig,axs= plt.subplots(1,3,figsize=(12,6))
    axs[0].set_title("Image")
    axs[1].set_title("FFT Power spectra")
    axs[2].set_title("FFT Phase spectra")
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    img0=axs[0].imshow(image,cmap='gray')
    fig.colorbar(img0,shrink=0.4)
    img1=axs[1].imshow(spectrum_modul,cmap='gray',vmin=-7.5,vmax=15)
    fig.colorbar(img1,shrink=0.4)
    img2=axs[2].imshow(spectrum_phase,cmap='twilight_shifted')
    fig.colorbar(img2,shrink=0.4)
    plt.tight_layout()


    
file_path = 'C:/UNIVERSIDAD/Repu/Workspace/nanoREPU2024/Quantum well/Mag3.67 cangle 159-200 CL37mm df193.05/100 frms 20230224 1840 STEM 27.2 nm HAADF 3.70 Mx Nano Diffraction_crop.tif'
s=hs.load(file_path)
s=hs.signals.Signal2D(plt.imread(file_path))
path = os.path.splitext(file_path)[0]
if not (os.path.exists(path)):
    os.mkdir(path)


pixel_size_pm=6.652

s.axes_manager[0].name = 'X'
s.axes_manager[1].name = 'Y'
s.axes_manager[0].scale = pixel_size_pm/1000
s.axes_manager[1].scale = pixel_size_pm/1000
s.axes_manager[0].units = 'nm'
s.axes_manager[1].units = 'nm'
s.metadata.General.title = ''
#s.plot(colorbar=False)
#ax=plt.gca()
#norm = mpl.colors.Normalize(vmin=np.min(s.data), vmax=np.max(s.data))
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="10%", pad=0.15)
#plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greys_r'),ax=ax, pad=.05, fraction=.1,cax=cax)
#plt.tight_layout()





inner_angle=159
outer_angle=200

det_image = 'C:/UNIVERSIDAD/Repu/Workspace/nanoREPU2024/Cangle 130-200mm 17.59.33 Scanning Acquire_1.tif'
det_image=hs.load(det_image)
#det_image=hs.signals.Signal2D(plt.imread(det_image))
det_image=hs.signals.Signal2D(det_image)
s_normalised = am.quant.detector_normalisation(s, det_image, inner_angle=inner_angle, outer_angle=outer_angle)
#s_normalised.plot(colorbar=False)
#ax=plt.gca()
#norm = mpl.colors.Normalize(vmin=np.min(s_normalised.data), vmax=np.max(s_normalised.data))
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="10%", pad=0.15)
#plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Greys_r'),ax=ax, pad=.05, fraction=.1,cax=cax)
#plt.tight_layout()


pixels_cropped=32
original_imag=s_normalised.data
sigma_est = np.mean(estimate_sigma(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]))
print(f'estimated noise standard deviation = {sigma_est}')
#fig, ax = plt.subplots(figsize=(12,8))
#ax.set_title('Original Image')
#ax.imshow(original_imag[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped], cmap='gray')
#ax.axis('off')




text='original_imag'


image=eval(text)[pixels_cropped:-pixels_cropped,pixels_cropped:-pixels_cropped]
high_pass=np.mean(np.abs(fft.ifft2((fft.fft2(image)==fft.fft2(image)[0,0])*fft.fft2(image))))+butterworth(image,0.005,True,order=2)
band_pass=butterworth(high_pass,0.08,False,order=4)


spectrum=fft.fftshift(fft.fft2(image))
spectrum_modul = np.log(np.abs(spectrum))
spectrum_phase = np.angle(spectrum)
fig,axs= plt.subplots(1,3,figsize=(12,6))
axs[0].set_title("Image")
axs[1].set_title("FFT Power spectra")
axs[2].set_title("FFT Phase spectra")
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
img0=axs[0].imshow(image,cmap='gray')
fig.colorbar(img0,shrink=0.4)
img1=axs[1].imshow(spectrum_modul,cmap='gray',vmin=-7.5,vmax=15)
fig.colorbar(img1,shrink=0.4)
img2=axs[2].imshow(spectrum_phase,cmap='twilight_shifted')
fig.colorbar(img2,shrink=0.4)
plt.tight_layout()



# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=1,
)


def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()
    
    

