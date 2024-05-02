import hyperspy.api as hs
import atomap.api as am
import matplotlib.pyplot as plt
import os, time
import numpy as np

from setting_image import open_file, imp, compare
from atomap_image import get_sublattice, intensity_map, intensity_map2, plot_lattice
from profile_image import fitt_muraki, composition_profile, plot_profile

st = time.time()
hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
plt.rcParams['figure.figsize'] = (8,8)


def make_lattice(path, image, optimal_separation, optimal_separation_d , dumbell):
    if (os.path.exists(path+'\\data.hdf5')):
        atom_lattice = am.load_atom_lattice_from_hdf5(path+'\\data.hdf5',construct_zone_axes=False)
    else:
        atom_lattice = get_sublattice(image, optimal_separation, optimal_separation_d , dumbell)
        try:
            atom_lattice.save(path+'\\data.hdf5', overwrite=True)
        except:
            print('Not able to save atom lattice')
    return atom_lattice

def make_intensity_map(path, atom_lattice, images):
    for image_name in images:
        
        if (os.path.exists(path+'\\im_A_'+image_name+'.npy')) & (os.path.exists(path+'\\im_B_'+image_name+'.npy')):
            intensity_A = np.load(path+'\\im_A_'+image_name+'.npy')
            intensity_B = np.load(path+'\\im_B_'+image_name+'.npy')
        else:
            intensity_A, intensity_B = intensity_map(globals()[image_name], atom_lattice)
            np.save(path+'\\im_A_'+image_name+'.npy',intensity_A)
            np.save(path+'\\im_B_'+image_name+'.npy',intensity_B)
    return intensity_B, intensity_A
            
def make_intensity_map2(path, atom_lattice, images):
    for image_name in images:
        
        if (os.path.exists(path+'\\im_B_'+image_name+'.npy')):
            print('Intensity map loaded')
            intensity_B = np.load(path+'\\im_B_'+image_name+'.npy')
        else:
            #intensity_A, intensity_B = intensity_map(globals()[image_name], atom_lattice)
            intensity_B = intensity_map2(globals()[image_name], atom_lattice)
            #np.save(path+'\\im_A_'+image_name+'.npy',intensity_A)
            np.save(path+'\\im_B_'+image_name+'.npy',intensity_B)
    return intensity_B


def find_optimal_pixel_sep(image, pixel_size_pm, gaussian = False):
    pixel_min_sep = round(340/pixel_size_pm/8)
    pixel_max_sep = round(400/pixel_size_pm/4)
    
    total_atoms = []
    pixel_separations = np.array(range (pixel_min_sep, pixel_max_sep))
    
    # First step: Assest symmetry
    neighbor_distances = []
    for optimal_separation in pixel_separations:
        print("Optimal separation: ", optimal_separation)
        try:
            atom_lattice, neighbor_distance = get_sublattice(image, optimal_separation, dumbell, find_lattice_error = True)
            neighbor_distances.append(neighbor_distance)
            total_atoms.append(np.array([atom_lattice.x_position, atom_lattice.y_position]).size)
        except:
            print("Not able to obtain consistent atom positions with this separation distance")
    neighbor_distances = np.array(neighbor_distances)
    plt.plot(pixel_separations, neighbor_distances[:,1],marker='o', color = 'navy', markersize= 13, linewidth=2)
    plt.xlabel('Minimum pixel separation [pixels]',fontsize = 16)
    plt.ylabel('First neighbord distances standard deviation [pm]',fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    index = np.argmin(neighbor_distances[:,1])
    optimal = pixel_separations[index]
    
    # Second step: Assess Gaussian Fitting
    if gaussian is True:
        gauss_pixel_separations = pixel_separations[max(0, index-4):min(len(pixel_separations), index+5)]
        gaussian_deviations = []
        for optimal_separation in gauss_pixel_separations:
            print("Optimal separation: ", optimal_separation)
            try:
                atom_lattice, gaussian_deviation = get_sublattice(image, optimal_separation, dumbell, find_gaussian_error = True)
                gaussian_deviations.append(gaussian_deviation)
            except:
                print("Not able to obtain consistent atom positions with this separation distance")
        gaussian_deviations = np.array(gaussian_deviations)
        sdv = np.sqrt(gaussian_deviations[:,0]**2 + gaussian_deviations[:,1] **2)
        plt.plot(gauss_pixel_separations, sdv,marker='o')
        plt.xlabel('Minimum pixel separation [pixels]')
        plt.legend(['Total atoms ratio [a.u.]', 'First neighbord distances standard deviation [pm]','Standard deviation error [pm]'],fontsize = 12)
        plt.grid()
        optimal = gauss_pixel_separations[np.argmin(gaussian_deviations[1])]
    else:
        print('Invalid finding')
        
        
    return optimal





global pixels_cropped, pixel_size_pm, inner_angle, outer_angle
pixel_size_pm=6.652 #3.326 #6.652 # pm
inner_angle=130
outer_angle=200
pixels_cropped=8

file_paths= open_file()
det_path= open_file()
det_image=hs.signals.Signal2D(plt.imread(det_path[0]))

global dumbell
dumbell = True

for file_path in file_paths:

    et = time.time()
    print(et - st)
    
    s=hs.load(file_path)
    #s = s.isig[500:1500, 500:3500]
    #s = s.isig[1000:2333, 0:4000]
    path = os.path.splitext(file_path)[0]
    if not (os.path.exists(path)):
        os.mkdir(path)

    SL = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL_pca = imp(s,pixels_cropped, pixel_size_pm, inner_angle,outer_angle)
    SL.scale(det_image)
    SL_pca.scale(det_image)
    SL_pca.image.data=SL.PCA(8)
    optimal_separation_d = 24
    #optimal_separation = find_optimal_pixel_sep(SL.image, SL.pixel_size_pm ,gaussian=False)
    optimal_separation = 9
    atom_lattice = make_lattice(path, SL.image, optimal_separation, optimal_separation_d, dumbell)
    
    
    # Intensity map
    pca_imag =  SL_pca.image.data
    images = ["pca_imag"]
    intensity_A, intensity_B = make_intensity_map(path, atom_lattice, images)

    
    #compare(original_imag , pca_imag, 'PCA - 8')
    #intensity_A, intensity_B = intensity_B, intensity_A
    #composition_profile(intensity_A, intensity_B, atom_lattice)
    
    print('end')

    