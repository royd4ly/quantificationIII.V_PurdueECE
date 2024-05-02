import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import atomap.api as am
hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()
plt.rcParams['figure.figsize'] = (7,7)


def Quality(s):
    
    S_MD = s.metadata
    S_OMD = s.original_metadata 
    
    x_ps_scal = s.axes_manager[0].scale
    x_ps_unit = s.axes_manager[0].units
    y_ps_scal = s.axes_manager[1].scale
    y_ps_unit = s.axes_manager[1].units
    #x_ps_scal = S_OMD.get_item('ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Scale')
    #x_ps_unit = S_OMD.get_item('ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup0.Units')
    #y_ps_scal = S_OMD.get_item('ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup1.Scale')
    #y_ps_unit = S_OMD.get_item('ImageList.TagGroup0.ImageData.Calibrations.Dimension.TagGroup1.Units')
    
    if (x_ps_scal == y_ps_scal) and (x_ps_unit == y_ps_unit):
        x_ps_unit = 1e-9 if x_ps_unit == "nm" else (1e-12 if x_ps_unit == "pm" else x_ps_unit)
        ps = x_ps_scal*x_ps_unit
    else:
        print("Both axis has not the same scale")
    
    dim_x = s.axes_manager[0].size
    dim_y = s.axes_manager[1].size
    mag = S_MD.get_item('Acquisition_instrument.TEM.magnification')
    dwell_time = S_MD.get_item('Acquisition_instrument.TEM.dwell_time')
    num_frames = S_OMD.get_item('ImageList.TagGroup0.ImageTags.DigiScan.Number_Summing_Frames')
    
    Collection_Angle = None
    for i in S_OMD.get_item('Detectors').keys():
        if S_OMD.get_item('Detectors.{}.DetectorName'.format(i)) == 'HAADF':
            Collection_Angle_i = S_OMD.get_item('Detectors.{}.CollectionAngleRange.begin'.format(i))
            Collection_Angle_f = S_OMD.get_item('Detectors.{}.CollectionAngleRange.end'.format(i))
    try:
        Collection_Angle = Collection_Angle_f - Collection_Angle_i
        return True if (Collection_Angle >= 0.05) & (ps <= 1e-9) & (mag >= 3.6e+6) else False
    except:
        print("No collection angle in metadata")




if __name__ == "__main__":
    s = hs.load(["20231205 1148 STEM 27.2 nm HAADF 3.70 Mx Nano Diffraction.emd"])[0]
    Quality(s)
    #s.plot(colorbar=False)    
    
    
    