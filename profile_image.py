import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from matplotlib.widgets import SpanSelector
from matplotlib import gridspec
from hyperspy.component import Component
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
import time
# Module to asses the interface profile and Muraki model fitting

# Muraki model stuff
def onselect(xmin, xmax):
    global x_pos 
    x_pos = np.array([xmin,xmax])
    
    
def mean_values(avg_intensity):
    count_binned=binned_statistic(avg_intensity,avg_intensity, 'count', bins=10)
    bin_centers=(count_binned[1][1:] + count_binned[1][:-1])/2
    mean_binned=binned_statistic(avg_intensity,avg_intensity, 'mean', bins=10)
    pos_peaks, _ = find_peaks(count_binned[0], height=0)
    pos_peaks_sorted=pos_peaks[np.argsort(count_binned[0][pos_peaks])]
    peaks_sorted=mean_binned[0][pos_peaks_sorted]
    
    n_lower_limit=3
    n_upper_limit=-2
    
    lower_limit,upper_limit=count_binned[1][n_lower_limit],count_binned[1][n_upper_limit]
    positions_l=np.where(avg_intensity<lower_limit)
    i_barriers=np.nanmean(avg_intensity[positions_l])
    positions_u=np.where(avg_intensity>upper_limit)
    i_quantum_well=np.nanmean(avg_intensity[positions_u])
    print('Mean intensity of the barriers: '+str(i_barriers))
    print('Mean intensity of the quantum well: '+str(i_quantum_well))
    return i_barriers, i_quantum_well



def fitt_muraki(intensity_map):
    class Muraki(Component):
        def __init__(self, parameter_1=1, parameter_2=2, parameter_3=3):
            Component.__init__(self, ('x0', 's', 'N'))
            self.x0.value = 1
            self.s.value = 0.5
            self.N.value = 5
            self.x0.bmin = 0
            self.x0.bmax = 1
            self.s.bmin = 0
            self.s.bmax = 1
            self.N.bmin = 0
            self.N.bmax = 50
        def function(self, x):
            x0 = self.x0.value
            s = self.s.value
            N = self.N.value
            return np.piecewise(x,[((x >= 1.0) & (x<= N)),x >= N],[lambda x : x0*(1.0 -s**x), lambda x: x0*(1 -s**x)*s**(x-N)])
    
    
    class Muraki2(Component):
        def __init__(self, parameter_1=1, parameter_2=2, parameter_3=3, parameter_4=4):
            Component.__init__(self, ('x0', 's1', 's2', 'N'))
            self.x0.value = 1.25
            self.s1.value = 0.5
            self.s2.value = 0.5
            self.N.value = 4
            self.x0.bmin = 0
            self.x0.bmax = 1
            self.s1.bmin = 0
            self.s1.bmax = 1
            self.s2.bmin = 0
            self.s2.bmax = 1
            self.N.bmin = 0
            self.N.bmax = 50
        def function(self, x):
            x0 = self.x0.value
            s1 = self.s1.value
            s2 = self.s2.value
            N = self.N.value
            return np.piecewise(x,[((x >= 1.0) & (x<= N)),x >= N],[lambda x : x0*(1.0 -s1**x), lambda x: x0*(1 -s2**x)*s2**(x-N)])

    def f(x,x0,s,N):
        return np.piecewise(x,[((x >= 1.0) & (x<= N)),x >= N],[lambda x : x0*(1.0 -s**x), lambda x: x0*(1 -s**x)*s**(x-N)])

    avg_intensity=np.nanmean(intensity_map[:,:,0],axis=1)
    i_barriers, i_quantum_well = mean_values(avg_intensity)
    
    
    normalized_array=(intensity_map-i_barriers)/(i_quantum_well-i_barriers)
    avg_norm=np.nanmean(normalized_array[:,:,0],axis=1)
    std_dev_norm=np.nanstd(normalized_array[:,:,0],axis=1)
    
    
    # Selection of the QW
    fig = plt.figure(figsize=(14, 8)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
    img1=ax0.plot(avg_norm,'*--')
    ax0.set(xlabel='Layer',ylabel='Average Composition')

    span = SpanSelector(
    ax0,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
    )
    time.sleep(3)
    ax1 = plt.subplot(gs[1])
    img2=ax1.scatter(intensity_map[:,:,1],intensity_map[:,:,2],s=20,c=normalized_array[:,:,0],cmap='jet',vmin=-0.25,vmax=1.25)
    fig.colorbar(img2,shrink=0.4,pad=0)
    ax1.axis('scaled')
    ax1.axis('off')
    ax1.set_ylim(ax1.get_ylim()[::-1]) 
    plt.tight_layout()
    plt.show(block=False)
    
    
    
    # Signal to fit with Muraki model
    muraki_positions=np.arange(x_pos[0]+1,x_pos[1]+1,dtype=int)
    muraki_signal=avg_norm[muraki_positions]
    std_dev=std_dev_norm[muraki_positions]
    sc=hs.signals.Signal1D(muraki_signal)
    print('Lower layer of the selection: '+str(muraki_positions[0]))
    print('Upper layer of the selection: '+str(muraki_positions[-1]))

    # 
    muraki_model = sc.create_model()
    muraki = Muraki()
    muraki_model.append(muraki)
    muraki_model.fit()
    muraki_model.print_current_values()

    x=np.arange(0,len(sc.data),dtype=float)
    y_pred=f(x,muraki.x0.value,muraki.s.value,muraki.N.value)
    r2_parameter=r2_score(sc.data[0::], y_pred[0::])    

    plt.figure()
    plt.plot(avg_axis[np.arange(0,len(avg_intensity))],avg_norm,'*--')
    plt.plot(avg_axis[x.astype(int)+muraki_positions[0]],y_pred,'-',color='red')
    plt.xlabel('Position [nm]')
    plt.ylabel('Average Composition')
    plt.minorticks_on()
    plot=plt.gca()
    label='$R^2 = $'+str(np.round(r2_parameter,3))
    at = AnchoredText(label, prop=dict(size=10), frameon=True, loc='upper right')
    at.patch.set(edgecolor='lightgray')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
    plot.add_artist(at)
    plt.show(block=False)


def composition_profile(intensity_A, intensity_B, atom_lattice):
    global avg_axis, avg_intensity, avg_std, avg_axis1, avg_intensity1, avg_std1
    # Parameters
    avg_intensity=np.nanmean(intensity_A[:,:,0],axis=1)
    avg_std=np.nanstd(intensity_A[:,:,0],axis=1)
    avg_axis=np.nanmean(intensity_A[:,:,2],axis=1)*atom_lattice.pixel_size
    avg_intensity1=np.nanmean(intensity_B[:,:,0],axis=1)
    avg_std1=np.nanstd(intensity_B[:,:,0],axis=1)
    avg_axis1=np.nanmean(intensity_B[:,:,2],axis=1)*atom_lattice.pixel_size
    
    plot_profile(intensity_A,intensity_B)

    fitt_muraki(intensity_A)
    
    

def plot_profile(intensity_A,intensity_B):

    # Create figure and axes with custom size
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [2.5, 2.5, 4]})
    
    # First subplot
    scatter1 = axs[0].scatter(intensity_A[:,:,1], intensity_A[:,:,2], s=20, c=intensity_A[:,:,0], cmap='gnuplot')
    #plt.colorbar(scatter1, ax=axs[0], shrink=0.5, pad=-0.18)
    axs[0].set_aspect('equal')
    axs[0].axis('off')
    axs[0].xaxis.tick_top()
    axs[0].yaxis.tick_left()
    axs[0].set_ylim(axs[0].get_ylim()[::-1]) 
    axs[0].set_title('Group III')
    
    # Second subplot
    scatter2 = axs[1].scatter(intensity_B[:,:,1], intensity_B[:,:,2], s=20, c=intensity_B[:,:,0], cmap='jet', vmin = np.nanmin(intensity_A[:,:,0])/1.01, vmax = 1.009*np.nanmax(intensity_A[:,:,0]))
    #plt.colorbar(scatter2, ax=axs[1], shrink=0.5, pad=-0.18)
    axs[1].set_aspect('equal')
    axs[1].axis('off')
    axs[1].xaxis.tick_top() 
    axs[1].yaxis.tick_left()
    axs[1].set_ylim(axs[1].get_ylim()[::-1]) 
    axs[1].set_title('Group V')
    
    # Third subplot
    axs[2].errorbar(avg_axis, avg_intensity, yerr=avg_std, ecolor='lightcoral', marker='.', fmt=':', capsize=5, alpha=0.75, mec='red', mfc='red', color='red', markersize=10)
    axs[2].fill_between(avg_axis, avg_intensity - avg_std, avg_intensity + avg_std, alpha=.25, color='lightcoral')
    axs[2].errorbar(avg_axis1, avg_intensity1, yerr=avg_std1, ecolor='darkgreen', marker='.', fmt=':', capsize=3, alpha=0.75, mec='skyblue', mfc='skyblue', color='skyblue', markersize=10)
    axs[2].fill_between(avg_axis1, avg_intensity1 - avg_std1, avg_intensity1 + avg_std1, alpha=.25, color='darkgreen')
    axs[2].set_xlabel('Position [nm]',fontsize = 16)
    axs[2].set_ylabel('Intensity',fontsize = 16)
    axs[2].minorticks_on()
    axs[2].grid(which='both', linestyle='--', color='gray', alpha=0.5)
    axs[2].set_title('Intensity vs Position')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the figure
    plt.show(block=False)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
