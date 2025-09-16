import os
import logging
import tempfile
import shutil

from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interpn
import skimage
import dbdicom as db
import vreg
import pydmr
from radiomics import featureextractor
from totalsegmentator.map_to_binary import class_map

datapath = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
maskpath = os.path.join(os.getcwd(), 'build', 'totseg', 'stage_1_segment')
measurepath = os.path.join(os.getcwd(), 'build', 'totseg', 'stage_3_measure')
os.makedirs(measurepath, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(measurepath, 'error.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


SHORT_NAME = {
    'kidney_left': 'LK',
    'kidney_right': 'RK',
    'kidneys_both': 'BK',
    'out_phase': 'op',
    'in_phase': 'ip',
    'fat': 'fi',
    'water': 'wi',
}

def shortname(longname):
    try:
        return SHORT_NAME[longname]
    except KeyError:
        return longname


biomarker_units = {
    'firstorder_Energy': 'Intensity^2 units',
    'firstorder_TotalEnergy': 'Intensity^2 units',
    'firstorder_Entropy': 'unitless',
    'firstorder_Kurtosis': 'unitless',
    'firstorder_Mean': 'Intensity units',
    'firstorder_Median': 'Intensity units',
    'firstorder_Minimum': 'Intensity units',
    'firstorder_Maximum': 'Intensity units',
    'firstorder_Skewness': 'unitless',
    'firstorder_StandardDeviation': 'Intensity units',
    'firstorder_Variance': 'Intensity^2 units',
    'firstorder_RootMeanSquared': 'Intensity units',
    'shape_VoxelVolume': 'mm^3',
    'shape_SurfaceArea': 'mm^2',
    'shape_SurfaceVolumeRatio': '1/mm',
    'shape_Compactness1': 'unitless',
    'shape_Compactness2': 'unitless',
    'shape_Sphericity': 'unitless',
    'shape_SphericalDisproportion': 'unitless',
    'shape_Maximum3DDiameter': 'mm',
    'shape_MajorAxisLength': 'mm',
    'shape_MinorAxisLength': 'mm',
    'shape_Elongation': 'unitless',
    'shape_Flatness': 'unitless',
    'glcm_Contrast': 'unitless',
    'glcm_Correlation': 'unitless',
    'glcm_DifferenceEntropy': 'unitless',
    'glcm_Id': 'unitless',
    'glcm_Idm': 'unitless',
    'glcm_Imc1': 'unitless',
    'glcm_Imc2': 'unitless',
    'glcm_InverseVariance': 'unitless',
}



def interpolate3d_isotropic(array, spacing, isotropic_spacing=None):

    if isotropic_spacing is None:
        isotropic_spacing = np.amin(spacing)

    # Get x, y, z coordinates for array
    nx = array.shape[0]
    ny = array.shape[1]
    nz = array.shape[2]
    Lx = (nx-1)*spacing[0]
    Ly = (ny-1)*spacing[1]
    Lz = (nz-1)*spacing[2]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    # Get x, y, z coordinates for isotropic array
    nxi = 1 + np.floor(Lx/isotropic_spacing)
    nyi = 1 + np.floor(Ly/isotropic_spacing)
    nzi = 1 + np.floor(Lz/isotropic_spacing)
    Lxi = (nxi-1)*isotropic_spacing
    Lyi = (nyi-1)*isotropic_spacing
    Lzi = (nzi-1)*isotropic_spacing
    xi = np.linspace(0, Lxi, nxi.astype(int))
    yi = np.linspace(0, Lyi, nyi.astype(int))
    zi = np.linspace(0, Lzi, nzi.astype(int))

    # Interpolate to isotropic
    ri = np.meshgrid(xi,yi,zi, indexing='ij')
    array = interpn((x,y,z), array, np.stack(ri, axis=-1))
    return array, isotropic_spacing


def volume_features(vol, roi):

    arr = vol.values

    # Scale array in the range [0,1] so it can be treated as mask
    # Motivation: the function is intended for mask arrays but this will make
    # sure the results are meaningful even if non-binary arrays are provided.
    max = np.amax(arr)
    min = np.amin(arr)
    arr -= min
    arr /= max-min

    # Add zeropadding at the boundary slices for masks that extend to the edge
    # Motivation: this could have some effect if surfaces are extracted - could create issues
    # if the values extend right up to the boundary of the slab.
    shape = list(arr.shape)
    shape[-1] = shape[-1] + 2*4
    array = np.zeros(shape)
    array[:,:,4:-4] = arr

    # Get voxel dimensions from the affine
    # We are assuming here the voxel dimensions are in mm.
    # If not the units provided with the return values are incorrect.
    spacing = vol.spacing
    voxel_volume = spacing[0]*spacing[1]*spacing[2]
    nr_of_voxels = np.count_nonzero(array > 0.5)
    volume = nr_of_voxels * voxel_volume

    # Surface properties - for now only extracting surface area
    try:
        # Note: this is smoothing the surface first - not tested in depth whether this is necessary or helpful.
        # It does appear to make a big difference on surface area so should be looked at more carefully.
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=0.5, step_size=1.0)
    except:
        # If a mask has too few points, smoothing can reduce the max to below 0.5. Use the midpoint in that case
        # Note this may work in general but 0.5 has been used for previous data collection so keep that as default
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=np.mean(smooth_array), step_size=1.0)
    surface_area = skimage.measure.mesh_surface_area(verts, faces)

    # Interpolate to isotropic for non-isotropic voxels
    # Motivation: this is required by the region_props function
    spacing = np.array(spacing)
    if np.amin(spacing) != np.amax(spacing):
        array, isotropic_spacing = interpolate3d_isotropic(array, spacing)
        isotropic_voxel_volume = isotropic_spacing**3
    else:
        isotropic_spacing = np.mean(spacing)
        isotropic_voxel_volume = voxel_volume

    # Get volume properties - mostly from region_props, except for compactness and depth
    array = np.round(array).astype(np.int16)
    region_props_3D = skimage.measure.regionprops(array)[0]

    # Calculate 'compactness' (our definition) - define as volume to surface ratio
    # expressed as a percentage of the volume-to-surface ration of an equivalent sphere.
    # The sphere is the most compact of all shapes, i.e. it has the largest volume to surface area ratio,
    # so this is guaranteed to be between 0 and 100%
    radius = region_props_3D['equivalent_diameter_area']*isotropic_spacing/2 # mm
    v2s = volume/surface_area # mm
    v2s_equivalent_sphere = radius/3 # mm
    compactness = 100 * v2s/v2s_equivalent_sphere # %

    # Fractional anisotropy - in analogy with FA in diffusion 
    m0 = region_props_3D['inertia_tensor_eigvals'][0]
    m1 = region_props_3D['inertia_tensor_eigvals'][1]
    m2 = region_props_3D['inertia_tensor_eigvals'][2]
    m = (m0 + m1 + m2)/3 # average moment of inertia (trace of the inertia tensor)
    FA = np.sqrt(3/2) * np.sqrt((m0-m)**2 + (m1-m)**2 + (m2-m)**2) / np.sqrt(m0**2 + m1**2 + m2**2)

    # Measure maximum depth (our definition)
    distance = ndi.distance_transform_edt(array)
    max_depth = np.amax(distance)

    # Adding a try/except around each line as some of these fail (math error) for masks with limited non-zero values

    roi = shortname(roi)
    data = {}
    try:
        data[f'{roi}-shape-ski_surface_area'] = [surface_area/100, f'Surface area ({roi})', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing surface area ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_volume'] = [volume/1000, f'Volume ({roi})', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Volume ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_bounding_box_volume'] = [region_props_3D['area_bbox']*isotropic_voxel_volume/1000, f'Bounding box volume ({roi})', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Bounding box volume ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_convex_hull_volume'] = [region_props_3D['area_convex']*isotropic_voxel_volume/1000, f'Convex hull volume ({roi})', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Convex hull volume ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_volume_of_holes'] = [(region_props_3D['area_filled']-region_props_3D['area'])*isotropic_voxel_volume/1000, f'Volume of holes ({roi})', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Volume of holes ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_extent'] = [region_props_3D['extent']*100, f'Extent ({roi})', '%', 'float']    # Percentage of bounding box filled
    except Exception as e:
        logging.error(f"Error computing Extent ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_solidity'] = [region_props_3D['solidity']*100, f'Solidity ({roi})', '%', 'float']   # Percentage of convex hull filled
    except Exception as e:
        logging.error(f"Error computing Solidity ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_compactness'] = [compactness, f'Compactness ({roi})', '%', 'float']
    except Exception as e:
        logging.error(f"Error computing Compactness ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_long_axis_length'] = [region_props_3D['axis_major_length']*isotropic_spacing/10, f'Long axis length ({roi})', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Long axis length ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_short_axis_length'] = [region_props_3D['axis_minor_length']*isotropic_spacing/10, f'Short axis length ({roi})', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Short axis length ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_equivalent_diameter'] = [region_props_3D['equivalent_diameter_area']*isotropic_spacing/10, f'Equivalent diameter ({roi})', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Equivalent diameter ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_maximum_depth'] = [max_depth*isotropic_spacing/10, f'Maximum depth ({roi})', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Maximum depth ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_primary_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][0]*isotropic_spacing**2/100, f'Primary moment of inertia ({roi})', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Primary moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_second_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][1]*isotropic_spacing**2/100, f'Second moment of inertia ({roi})', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Second moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_third_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][2]*isotropic_spacing**2/100, f'Third moment of inertia ({roi})', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Third moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_mean_moment_of_inertia'] = [m*isotropic_spacing**2/100, f'Mean moment of inertia ({roi})', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Mean moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_fractional_anisotropy_of_inertia'] = [100*FA, f'Fractional anisotropy of inertia ({roi})', '%', 'float']
    except Exception as e:
        logging.error(f"Error computing Fractional anisotropy of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape-ski_volume_qc'] = [region_props_3D['area']*isotropic_voxel_volume/1000, f'Volume QC ({roi})', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Volume QC ({roi}): {e}")
    # Taking this out for now - computation uses > 32GB of memory for large masks
    # data[f'{roi}_ski_longest_caliper_diameter'] = [region_props_3D['feret_diameter_max']*isotropic_spacing/10, f'Longest caliper diameter ({roi})', 'cm', 'float']

    return data




       

def radiomics_shape_features(roi_vol, roi):
    tmp = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp, exist_ok=True)
    roi_file = os.path.join(tmp, 'roi.nii.gz')
    img_file = os.path.join(tmp, 'img.nii.gz') # dummy
    vreg.write_nifti(roi_vol, roi_file)
    vreg.write_nifti(roi_vol, img_file)
    # All features for water
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    result = extractor.execute(img_file, roi_file)
    shutil.rmtree(tmp)
    # Format return value
    rval = {}
    for p, v in result.items():
        if p[:8]=='original':
            name = shortname(roi) + '-' + p.replace('original_shape_', 'shape-')
            vals = [float(v), name, 'unit', 'float']
            rval[name] = vals
    return rval


def radiomics_texture_features(roi_vol, img_vol, roi, img):
    tmp = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp, exist_ok=True)
    roi_file = os.path.join(tmp, 'roi.nii.gz')
    img_file = os.path.join(tmp, 'img.nii.gz')
    print('radiomics texture', roi)
    # Downsample large ROIs to avoid memory over
    # TODO: Not enough for large regions - still RAM issue
    roi_vol_box = roi_vol.crop(mask=roi_vol) # some edits in vreg. Check
    img_vol_box = img_vol.crop(mask=roi_vol)
    # roi_vol_box = roi_vol_box.resample(5.0)
    # img_vol_box = img_vol_box.resample(5.0)
    vreg.write_nifti(roi_vol_box, roi_file)
    vreg.write_nifti(img_vol_box, img_file)
    # All features for water
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    # TODO: try without first order
    classes = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    for cl in classes:
        extractor.enableFeatureClassByName(cl)
    # extractor.enableImageTypeByName('Wavelet')
    # extractor.enableImageTypeByName('LoG', {'sigma': [1.0, 1.0, 1.0]}) 
    # extractor.enableImageTypeByName('Gradient')
    result = extractor.execute(img_file, roi_file)
    shutil.rmtree(tmp)
    # Format return value
    rval = {}
    for p, v in result.items():
        if p[:8] == 'original':
            name = shortname(roi) + '-' + shortname(img) + '-' 
            for cl in classes:
                if cl in p:
                    name += p.replace(f'original_{cl}_', f'{cl}-')
                    break
            vals = [float(v), name, 'unit', 'float']
            rval[name] = vals
    return rval



def bari():

    sitedatapath = os.path.join(datapath, "Patients", "Bari") 
    sitemaskpath = os.path.join(maskpath, "Patients", "Bari")
    sitemeasurepath = os.path.join(measurepath, "Patients", "Bari")

    all_mask_series = db.series(sitemaskpath)
    for mask_series in tqdm(all_mask_series, desc='Extracting metrics'):
        vol = db.volume(mask_series)
        patient, series = mask_series[1], mask_series[3][0]
        dir = os.path.join(sitemeasurepath, patient)
        os.makedirs(dir, exist_ok=True)

        class_maps = {
            'totseg': class_map['total_mr'],
            'nnunet': {1: "kidney_left", 2: "kidney_right"},
            'unetr': {1: "kidney_left", 2: "kidney_right"},
        }
        model = series.split('_')[-1]

        # Loop over the classes
        for idx, roi in class_maps[model].items():
            mask = (vol.values==idx).astype(np.float32)
            if np.sum(mask) == 0:
                continue
            dmr_file = os.path.join(dir, f"{series}_{roi}")
            if os.path.exists(f'{dmr_file}.dmr.zip'):
                continue
            roi_vol = vreg.volume(mask, vol.affine)
            dmr = {'data':{}, 'pars':{}}

            # Get skimage features
            results = volume_features(roi_vol, roi)
            dmr['data'] = dmr['data'] | {p: v[1:] for p, v in results.items()}
            dmr['pars'] = dmr['pars'] | {(patient, 'Baseline', p): v[0] for p, v in results.items()}

            # Get radiomics shape features
            results = radiomics_shape_features(roi_vol, roi)
            dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
            dmr['pars'] = dmr['pars'] | {(patient, 'Baseline', p): v[0] for p, v in results.items()}

            # Get radiomics texture features
            if roi in ['kidney_left', 'kidney_right']: # computational issues with larger ROIs. Exclude for now
                for img in ['out_phase', 'in_phase', 'fat', 'water']:
                    img_series = [sitedatapath, patient, 'Baseline', series[:-len(model)] + img]
                    try:
                        img_vol = db.volume(img_series)
                    except:
                        pass # series does not exist
                    else:
                        results = radiomics_texture_features(roi_vol, img_vol, roi, img)
                        dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
                        dmr['pars'] = dmr['pars'] | {(patient, 'Baseline', p): v[0] for p, v in results.items()}

            # Write results to file
            pydmr.write(dmr_file, dmr)

        # Both kidneys texture
        class_index = {roi:idx for idx,roi in class_maps[model].items()}
        lk_mask = (vol.values==class_index['kidney_left']).astype(np.float32)
        rk_mask = (vol.values==class_index['kidney_right']).astype(np.float32)
        roi = 'kidneys_both'
        mask = lk_mask + rk_mask
        if np.sum(mask) == 0:
            continue
        dmr_file = os.path.join(dir, f"{series}_{roi}")
        if os.path.exists(dmr_file):
            continue
        roi_vol = vreg.volume(mask, vol.affine)
        dmr = {'data':{}, 'pars':{}}

        # Get radiomics texture features
        for img in ['out_phase', 'in_phase', 'fat', 'water']:
            img_series = [sitedatapath, patient, 'Baseline', series[:-len(model)] + img]
            try:
                img_vol = db.volume(img_series)
            except:
                pass # series does not exist
            else:
                results = radiomics_texture_features(roi_vol, img_vol, roi, img)
                dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
                dmr['pars'] = dmr['pars'] | {(patient, 'Baseline', p): v[0] for p, v in results.items()}

        # Write results to file
        pydmr.write(dmr_file, dmr)


def bari_concat():
    sitemeasurepath = os.path.join(measurepath, "Patients", "Bari")
    # Concatenate all dmr files of each subject
    patients = [f.path for f in os.scandir(sitemeasurepath) if f.is_dir()]
    for patient in patients:
        dir = os.path.join(sitemeasurepath, patient)
        dmr_files = [f for f in os.listdir(dir) if f.endswith('.dmr.zip')]
        dmr_files = [os.path.join(dir, f) for f in dmr_files]
        dmr_file = os.path.join(sitemeasurepath, f'{patient}_results')
        pydmr.concat(dmr_files, dmr_file)

