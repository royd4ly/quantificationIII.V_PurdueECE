import atomap.initial_position_finding as ipf
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import atomap.api as am

# Module to make the atomap image processing: get subalttice and intensity maps



def lattice_error(sublattice):
    positions = np.array([sublattice.x_position, sublattice.y_position]).swapaxes(0, 1)
      
    # cKDTree
    tree = cKDTree(positions)
    distances, indices = tree.query(positions, k=2)  # k=2 for 1st neighbord
    

    nearest_neighbor_distances = distances[:, 1]
    

    mean_distance = np.mean(nearest_neighbor_distances)
    std_dev_distance = np.std(nearest_neighbor_distances)
    
    #print("Distancias a los vecinos más cercanos:", nearest_neighbor_distances)
    print("Distancia promedio:", mean_distance)
    print("Desviación estándar de las distancias:", std_dev_distance)
    
    neighbor_distances = [mean_distance, std_dev_distance]
    return neighbor_distances
    

def gaussian_error(sublattice):
    sublattice.refine_atom_positions_using_2d_gaussian()
    atoms = sublattice.atom_list
    sx,sy = [],[]
    for i in range(np.array(atoms).size):
        sx.append(atoms[i].sigma_x)
        sy.append(atoms[i].sigma_y)
    error_sx = np.std(sx)
    error_sy = np.std(sy)
    gaussian_deviation = error_sx, error_sy
    return gaussian_deviation

def get_sublattice(s_normalised, optimal_separation, optimal_separation_d = 5 , dumbell = False,
                   find_lattice_error = False, find_gaussian_error = False):
    
    atom_positions = am.get_atom_positions(s_normalised, optimal_separation,
                                           pca=True,subtract_background=True, normalize_intensity=True)
    sublattice = am.Sublattice(atom_positions, s_normalised)
    sublattice.find_nearest_neighbors()
    sublattice.refine_atom_positions_using_center_of_mass()
    #sublattice.refine_atom_positions_using_2d_gaussian()
    #sublattice.get_atom_list_on_image(markersize=5).plot()
    
    
    if dumbell is True:
        dumbbell_vector = ipf.find_dumbbell_vector(atom_positions)
        dumbbell_positions = am.get_atom_positions(s_normalised, optimal_separation_d,pca=True,subtract_background=True, normalize_intensity=True)
        sublattice = am.Sublattice(dumbbell_positions, s_normalised)
        #sublattice.get_atom_list_on_image(markersize=5).plot()

        # Dumbell recognition
        dumbbell_positions = np.asarray(dumbbell_positions)
        dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(s_normalised, dumbbell_positions, dumbbell_vector)
        dumbbell_lattice.pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.sublattice_list[0].pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.sublattice_list[1].pixel_size=s_normalised.axes_manager[0].scale
        dumbbell_lattice.units=s_normalised.axes_manager[0].units
        dumbbell_lattice.sublattice_list[0].units=s_normalised.axes_manager[0].units
        dumbbell_lattice.sublattice_list[1].units=s_normalised.axes_manager[0].units
    else:
        if (find_lattice_error is True):   
            print("Finding lattice error")
            neighbor_distances = lattice_error(sublattice)
            return sublattice, neighbor_distances
        
        elif (find_gaussian_error is True):
            print('Finding gaussian error')
            gaussian_deviation = gaussian_error(sublattice)
            return sublattice, gaussian_deviation
        
        else:    
            sublattice.refine_atom_positions_using_2d_gaussian()
            sublattice.get_atom_list_on_image(markersize=5).plot()
            return sublattice
    
    
    
    return dumbbell_lattice

def intensity_map(image, atom_lattice):
    sublattice_A=atom_lattice.sublattice_list[1]
    sublattice_B=atom_lattice.sublattice_list[0]
    atom_lattice.units=atom_lattice.sublattice_list[0].units
    atom_lattice.pixel_size=atom_lattice.sublattice_list[0].pixel_size
    
    # Intensity map of A 
    sublattice_A.original_image= image
    sublattice_A.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_A._pixel_separation = sublattice_A._get_pixel_separation()
    sublattice_A._make_translation_symmetry()
    if ((0,0) in sublattice_A.zones_axis_average_distances):
        index=sublattice_A.zones_axis_average_distances.index((0,0))
        sublattice_A.zones_axis_average_distances.remove(sublattice_A.zones_axis_average_distances[index])
        sublattice_A.zones_axis_average_distances_names.remove(sublattice_A.zones_axis_average_distances_names[index])
    sublattice_A._generate_all_atom_plane_list(atom_plane_tolerance=0.5)
    sublattice_A._sort_atom_planes_by_zone_vector()
    sublattice_A._remove_bad_zone_vectors()
    
    
    direction=2

    zone_vector = sublattice_A.zones_axis_average_distances[direction]
    atom_planes = sublattice_A.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_A.get_atom_planes_on_image(atom_planes)
    
    # Plot directions
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show(block=False)
    #######

    sublattice_A.find_sublattice_intensity_from_masked_image(sublattice_A.original_image,radius=7)
    zone_axis_A = sublattice_A.zones_axis_average_distances[direction]
    atom_plane_list_A = sublattice_A.atom_planes_by_zone_vector[zone_axis_A]
    intensity_A=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_A)):
        atomplane=atom_plane_list_A[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_A.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_A_array = np.zeros([len(intensity_A),len(max(intensity_A,key = lambda x: len(x)))])
    intensity_A_array[:] = np.nan
    for i,j in enumerate(intensity_A):
        intensity_A_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_A=np.stack((intensity_A_array,x_values_array,y_values_array),axis=2)

    # Intnsity of B sublattice
    
    sublattice_B.original_image=image
    sublattice_B.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_B._pixel_separation = sublattice_B._get_pixel_separation()
    sublattice_B._make_translation_symmetry()
    if ((0,0) in sublattice_B.zones_axis_average_distances):
        index=sublattice_B.zones_axis_average_distances.index((0,0))
        sublattice_B.zones_axis_average_distances.remove(sublattice_B.zones_axis_average_distances[index])
        sublattice_B.zones_axis_average_distances_names.remove(sublattice_B.zones_axis_average_distances_names[index])
    sublattice_B._generate_all_atom_plane_list(atom_plane_tolerance=0.5)
    sublattice_B._sort_atom_planes_by_zone_vector()
    sublattice_B._remove_bad_zone_vectors()
    
    direction=2

    zone_vector = sublattice_B.zones_axis_average_distances[direction]
    atom_planes = sublattice_B.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_B.get_atom_planes_on_image(atom_planes)
    

    ##  Plot axis
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show(block=False)
    ######


    
    sublattice_B.find_sublattice_intensity_from_masked_image(sublattice_B.original_image,radius=5)
    zone_axis_B = sublattice_B.zones_axis_average_distances[direction]
    atom_plane_list_B = sublattice_B.atom_planes_by_zone_vector[zone_axis_B]
    intensity_B=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_B)):
        atomplane=atom_plane_list_B[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_B.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_B_array = np.zeros([len(intensity_B),len(max(intensity_B,key = lambda x: len(x)))])
    intensity_B_array[:] = np.nan
    for i,j in enumerate(intensity_B):
        intensity_B_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_B=np.stack((intensity_B_array,x_values_array,y_values_array),axis=2)
    
    return intensity_A, intensity_B


def intensity_map2(image, atom_lattice):
    sublattice_B=atom_lattice
    atom_lattice.units=atom_lattice.units
    atom_lattice.pixel_size=atom_lattice.pixel_size
    


    # Intnsity of B sublattice
    
    sublattice_B.original_image=image
    sublattice_B.find_nearest_neighbors(nearest_neighbors=15)
    sublattice_B._pixel_separation = sublattice_B._get_pixel_separation()
    sublattice_B._make_translation_symmetry()
    if ((0,0) in sublattice_B.zones_axis_average_distances):
        index=sublattice_B.zones_axis_average_distances.index((0,0))
        sublattice_B.zones_axis_average_distances.remove(sublattice_B.zones_axis_average_distances[index])
        sublattice_B.zones_axis_average_distances_names.remove(sublattice_B.zones_axis_average_distances_names[index])
    sublattice_B._generate_all_atom_plane_list(atom_plane_tolerance=1)
    sublattice_B._sort_atom_planes_by_zone_vector()
    sublattice_B._remove_bad_zone_vectors()
    
    direction=0

    zone_vector = sublattice_B.zones_axis_average_distances[direction]
    atom_planes = sublattice_B.atom_planes_by_zone_vector[zone_vector]
    zone_axis = sublattice_B.get_atom_planes_on_image(atom_planes)
    

    ##  Plot axis
    zone_axis.plot()
    ax = plt.gca()
    fig=plt.gcf()
    fig.set_size_inches((10,10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    ######


    
    sublattice_B.find_sublattice_intensity_from_masked_image(sublattice_B.original_image,radius=5)
    zone_axis_B = sublattice_B.zones_axis_average_distances[direction]
    atom_plane_list_B = sublattice_B.atom_planes_by_zone_vector[zone_axis_B]
    intensity_B=[]
    x_values=[]
    y_values=[]
    for i in range(0,len(atom_plane_list_B)):
        atomplane=atom_plane_list_B[i]
        plane_intensity=[]
        x_values_plane=[]
        y_values_plane=[]
        for j in range(0, len(atomplane.atom_list)):
            atom=atomplane.atom_list[j]
            x_pos,y_pos=atom.get_pixel_position()
            intensity=atom.intensity_mask
            plane_intensity.append(intensity)
            x_values_plane.append(x_pos)
            y_values_plane.append(y_pos)
        intensity_B.append(plane_intensity)
        x_values.append(x_values_plane)
        y_values.append(y_values_plane)
    
    intensity_B_array = np.zeros([len(intensity_B),len(max(intensity_B,key = lambda x: len(x)))])
    intensity_B_array[:] = np.nan
    for i,j in enumerate(intensity_B):
        intensity_B_array[i][0:len(j)] = j
    
    x_values_array = np.zeros([len(x_values),len(max(x_values,key = lambda x: len(x)))])
    x_values_array[:] = np.nan
    for i,j in enumerate(x_values):
        x_values_array[i][0:len(j)] = j
    
    y_values_array = np.zeros([len(y_values),len(max(y_values,key = lambda x: len(x)))])
    y_values_array[:] = np.nan
    for i,j in enumerate(y_values):
        y_values_array[i][0:len(j)] = j
        
    intensity_B=np.stack((intensity_B_array,x_values_array,y_values_array),axis=2)
    
    return intensity_B




def plot_lattice(atom_lattice):
    atom_lattice.get_atom_list_on_image(markersize=2).plot()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches((8,8))
