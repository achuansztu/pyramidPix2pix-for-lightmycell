"""
The following is the evaluation method for the France-BioImaging Light My Cells Challenge.

It is meant to run within a container.
This will start the evaluation, reads from ./test/input and outputs to ./test/output
To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-evaluation-phase-1 | gzip -c > example-evaluation-phase-1.tar.gz
"""
import json
from multiprocessing import Pool
from os.path import isfile
from pathlib import Path
from pprint import pformat, pprint
import os
import numpy as np
from scipy.stats import pearsonr
from tifffile import imread
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pandas as pd

INPUT_DIRECTORY = Path("/input")
print(" INPUT_PATH IS " + str(INPUT_DIRECTORY))
os.system("ls -lh   " + str(INPUT_DIRECTORY))

OUTPUT_DIRECTORY = Path("/output")
print(" OUTPUT IS " + str(OUTPUT_DIRECTORY))
os.system("ls -lh " + str(OUTPUT_DIRECTORY))

GROUND_TRUTH_DIRECTORY = Path("ground_truth")
print(" GROUND_TRUTH_DIRECTORY IS  " + str(GROUND_TRUTH_DIRECTORY))
os.system("ls -lh " + str(GROUND_TRUTH_DIRECTORY))


def get_img_metrics(organelle, img_gt, img_pred):
    '''
    Calculate image metrics for a given organelle.

    Parameters:
    - organelle (str): The type of organelle.
    - img_gt (np.ndarray): Ground truth image.
    - img_pred (np.ndarray): Predicted image.

    Returns:
    Dict[str, float]: Dictionary containing calculated metrics
    for the wanted organnelle.
    '''

    metrics_results = {}

    if img_gt is None :
        print("Image GT is None")
        return metrics_results

    if img_pred is None:
        raise ValueError("ERROR None prediction image")

    if img_gt.shape != img_pred.shape: # NOT SAME SHAPE
        print(f"{organelle} : ERROR SHAPES are not equal ! GT shape = {img_gt.shape} and Pred shape = {img_pred.shape}")
        raise ValueError(f"ERROR GT and predictions shapes are not equal ! \n \
                           GT shape = {img_gt.shape} and Pred shape = {img_pred.shape}")

    # perform percentile normalization
    try:
        img_gt = percentile_normalization(img_gt)
    except Exception as e:
        print(" --> ERROR NORMALIZATION GT ")
        print(e)
        return metrics_results

    try:
        img_pred = percentile_normalization(img_pred)
    except Exception as e:
        print(" --> ERROR NORMALIZATION PRED ")
        print(e)
        return metrics_results

    PCC, _ = pearsonr(img_gt.ravel(), img_pred.ravel() )

    SSIM = ssim(img_gt,
                img_pred,
                data_range=np.maximum(img_pred.max() - img_pred.min(),
                                      img_gt.max() - img_gt.min()))

    metrics_results['PCC'] = 0 if np.isnan(PCC) else PCC
    metrics_results['SSIM'] = SSIM

    if any(x in organelle for x in ["nucleus", "mitochondria"]):

        MAE = mean_absolute_error(img_gt, img_pred)
        e_dist = euclidean_distances(img_gt.ravel().reshape(1, -1),
                                     img_pred.ravel().reshape(1, -1))[0, 0]
        c_dist = np.abs(cosine_distances(img_gt.ravel().reshape(1, -1),
                                         img_pred.ravel().reshape(1, -1))[0, 0])

        metrics_results['MAE'] = MAE
        metrics_results['E_dist'] = e_dist
        metrics_results['C_dist'] = c_dist

    print(metrics_results)
    return metrics_results


def get_all_locations(job):

    job_pk = job["pk"]
    locations = {
        'job_pk' : job_pk,
        'inputs': {},
        'outputs': {
            'nucleus-fluorescence-ome-tiff': {},
            'mitochondria-fluorescence-ome-tiff': {},
            'tubulin-fluorescence-ome-tiff': {},
            'actin-fluorescence-ome-tiff': {}
        }
    }

    for value in job["inputs"]:
        if value["interface"]["slug"] == 'organelles-transmitted-light-ome-tiff':

            locations['inputs'] = {
                'relative_path': value["interface"]["relative_path"],
                'name' : value["image"]["name"],
                'pk': value["image"]["pk"]
            }
    for value in job["outputs"]:
        if value["interface"]["slug"] in locations['outputs']:
            relative_path = value["interface"]["relative_path"]
            image_name = value["image"]["pk"] + '.tiff'

            locations['outputs'][value["interface"]["slug"]] = INPUT_DIRECTORY / job_pk / "output" / relative_path / image_name

    return locations

def read_zandtl():

    csv_file = pd.read_csv(GROUND_TRUTH_DIRECTORY / "BESTZZ_Phase_1.csv", sep=',')
    csv_file.head()
    ztl={}
    for _, name in csv_file.iterrows():  # name,modality,image_z,best_z
        ztl[name['name']] = [name['modality'], int(name['image_z']), int(name['best_z'])]
    print(ztl)
    return ztl


def calcul_mean_organelles(metrics):

    mean_organelle={}
    for elet in metrics['results'] :
        for organelle in ["nucleus", "mitochondria", "actin", "tubulin"]:
            if organelle in elet and len(elet[organelle]) > 0:
                # not 'actin': {}
                if organelle not in mean_organelle:
                    mean_organelle[organelle] = {}

                for method in metrics_methods[organelle]:
                    if method not in mean_organelle[organelle]:
                        mean_organelle[organelle][method] = []
                    mean_organelle[organelle][method].append(elet[organelle][method])

    metrics_mean = {}
    for organelle in mean_organelle:
        metrics_mean[organelle] = {}
        for method in mean_organelle[organelle]:
            metrics_mean[organelle][method] = np.mean(mean_organelle[organelle][method])

    return metrics_mean


def get_original_names(predictions):
    names = {}
    for job in predictions:
        names[job["inputs"][0]['image']['pk']+'.tiff'] = job["inputs"][0]['image']['name']
    return names

def calcul_mean_tl(metrics, ztl, image_names):

    metrics_tl = {}
    for tl in ['BF', 'DIC', 'PC']:
        metrics_tl[tl] = {}

        mean_organelle = {}
        for organelle in ["nucleus", "mitochondria", "actin", "tubulin"]:
            metrics_results = {}
            for method in metrics_methods[organelle]:
                metrics_results[method] = []

            for elet in metrics['results'] :
                image_name = image_names[elet['image_name']]
                modality = ztl[image_name][0]

                if modality == tl and organelle in elet:
                    if len(elet[organelle]) > 0:  # not 'actin': {}
                        for method in metrics_methods[organelle]:
                            metrics_results[method].append(elet[organelle][method])

            mean_organelle[organelle] = {}
            for method in metrics_methods[organelle]:
                if len(metrics_results[method]) != 0:
                    mean_organelle[organelle][method] = np.mean(metrics_results[method])
                else:
                    mean_organelle[organelle][method] = np.NAN

        metrics_tl[tl] = mean_organelle
    return metrics_tl


def calcul_mean_z(metrics, ztl, image_names):
    
    metrics_z = {}
    for elet in metrics['results'] :
        image_name  = image_names[elet['image_name']]
        z_deviation = abs(ztl[image_name][1]-ztl[image_name][2])
        
        for organelle in ["nucleus", "mitochondria", "actin", "tubulin"]:
            if organelle in elet and len(elet[organelle]) > 0:  # not 'actin': {}
                if organelle not in metrics_z:
                    metrics_z[organelle] = {}
                if z_deviation not in metrics_z[organelle]:
                    metrics_z[organelle][z_deviation] = {}

                for method in metrics_methods[organelle]:
                    if method not in metrics_z[organelle][z_deviation]:
                        metrics_z[organelle][z_deviation][method] = []
                    metrics_z[organelle][z_deviation][method].append(elet[organelle][method])

    metrics_z_mean = {}
    for organelle in metrics_z:
        metrics_z_mean[organelle] = {}
        for z_deviation in metrics_z[organelle]:
            metrics_z_mean[organelle][z_deviation] = {}
            for method in metrics_z[organelle][z_deviation]:
                metrics_z_mean[organelle][z_deviation][method] = np.mean(metrics_z[organelle][z_deviation][method])

    return metrics_z_mean


metrics_methods = {}
metrics_methods["nucleus"] = ['MAE', 'PCC', 'SSIM', 'E_dist', 'C_dist']
metrics_methods["mitochondria"] = ['MAE', 'PCC', 'SSIM', 'E_dist', 'C_dist']
metrics_methods["actin"] = ['PCC', 'SSIM']
metrics_methods["tubulin"] = ['PCC', 'SSIM']


def main():
    '''
    Main function for the evaluation process.
    It processes each algorithm (=job) for this submission.
    then it reads the predictions.json (in the input folder) and
     starts a number of process workers (using multiprocessing)
    to compute the metrics between the prediction and ground truth images.
    Finally, it writes the metrics of each image into a json file (output folder).
    '''
    print(" START ")
    print_inputs()

    metrics = {}
    print("READ PREDICTIONS")
    predictions = read_predictions()

    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)
    print(metrics)

    # AGGREGATE METRICS
    metrics['agregate'] = calcul_mean_organelles(metrics)

    ztl=read_zandtl()

    #   TRANSMITTED LIGHT AGGREGATE
    image_names = get_original_names(predictions)
    metrics['transmitted_light'] = calcul_mean_tl(metrics,ztl,image_names)

    # Z AGGREGATE
    metrics['z_deviation'] = calcul_mean_z(metrics,ztl,image_names)

    print("METRICS OK ")
    # # Make sure to save the metrics
    write_metrics(metrics=metrics)

    print("METRICS written")
    return 0


def process(job):
    """
    Process a single algorithm job.

    Parameters:
    - job (Dict[str, Any]): Job information.

    Returns:
    Dict[str, Union[float, Dict]]: Results of the processing.
    """

    job_pk = str(job["pk"])
    print(job_pk + " -> Processing:")

    # get the input image GC id/name
    print(job_pk + " -> FIND PRED LOCATION")

    all_loc_dict = get_all_locations(job)

    input_TL_image_name = all_loc_dict['inputs']['pk'] + '.tiff'  # ONLY NAME + EXTENSION
    print(job_pk+ " -> input_TL_image_name="+input_TL_image_name)

    nucleus_pred_location = all_loc_dict['outputs']['nucleus-fluorescence-ome-tiff']  # PATH + FILENAME + EXTENSION
    print(job_pk+ " -> nucleus_pred_location=" + str(nucleus_pred_location))

    mitochondria_pred_location = all_loc_dict['outputs']['mitochondria-fluorescence-ome-tiff']
    print(job_pk+ " -> mitochondria_pred_location=" + str(mitochondria_pred_location))

    tubulin_pred_location = all_loc_dict['outputs']['tubulin-fluorescence-ome-tiff']
    print(job_pk + " -> tubulin_pred_location=" + str(tubulin_pred_location))

    actin_pred_location = all_loc_dict['outputs']['actin-fluorescence-ome-tiff']
    print(job_pk + " -> actin_pred_location=" + str(actin_pred_location))

    print(job_pk + " -> LOAD PRED IMAGES")
    # Load the predictions
    nucleus_pred = load_image_file(filename=nucleus_pred_location)
    mitochondria_pred = load_image_file(filename=mitochondria_pred_location)
    tubulin_pred = load_image_file(filename=tubulin_pred_location)
    actin_pred = load_image_file(filename=actin_pred_location)

    print(job_pk + " -> CHECK if NO missing pred")
    # check if there is no missing prediction
    if any(x is None for x in [nucleus_pred, mitochondria_pred, tubulin_pred, actin_pred]):
        raise ValueError(job_pk + " ->  ERROR: MISSING PREDCTIONS FILES ")

    # Load and read the ground truth
    gt_path = GROUND_TRUTH_DIRECTORY /"images"
    print(job_pk + " -> LOAD GT")
    nucleus_gt = load_image_file(filename=gt_path / "nucleus-fluorescence-ome-tiff" / input_TL_image_name)
    mitochondria_gt = load_image_file(filename=gt_path / "mitochondria-fluorescence-ome-tiff" / input_TL_image_name)
    tubulin_gt = load_image_file(filename=gt_path / "tubulin-fluorescence-ome-tiff" / input_TL_image_name)
    actin_gt = load_image_file(filename=gt_path / "actin-fluorescence-ome-tiff" / input_TL_image_name)

    print(job_pk + " -> METRICS CALCUL START")
    # Calculate and group the metrics by comparing the ground truth to the actual results
    results = {}
    results["job_name"] = all_loc_dict['job_pk']
    results["image_name"] = input_TL_image_name
    results["nucleus"] = get_img_metrics("nucleus", nucleus_gt, nucleus_pred)
    results["mitochondria"] = get_img_metrics("mitochondria", mitochondria_gt, mitochondria_pred)
    results["tubulin"] = get_img_metrics("tubulin", tubulin_gt, tubulin_pred)
    results["actin"] = get_img_metrics("actin", actin_gt, actin_pred)

    print(job_pk + " -> METRICS "+str(results))
    return results


def load_image_file(*,  filename):
    """
    Load an image file.

    Parameters:
    - filename (Path): Path to the image file.

    Returns:
    Optional[np.ndarray]: Loaded image as a NumPy array, or None if the file is not found.
    """
    if not isfile(filename):
        print(" MISSS "+str(filename))
        return None

    print(" FOUND "+str(filename))
    image = imread(filename)
    return image


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    '''
    Compute a percentile normalization for the given image.

    Parameters:
    - image (array): array of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed. 
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    '''

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_p = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        print(f"Same min {low_p} and high {high_p}, image may be empty")
    else:
        img_norm = (image - low_p) / (high_p - low_p)

    return img_norm


def read_predictions():
    """
    Reads the prediction file to get the location of the users' predictions.

    Returns:
    dict: Parsed JSON content from the predictions file.
    """
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def print_inputs():
    """
    Just for convenience,
    Prints information about input files in the logs.

    Returns:
    None
    """
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    print(input_files)
    print("")


def write_metrics(*, metrics):
    """
    Writes a JSON document used for ranking results on the leaderboard.

    Parameters:
    - metrics (dict): Metrics to be written to the document.

    Returns:
    None
    """
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))

    print(" CHECK OUTPUT IS   "+str(OUTPUT_DIRECTORY))
    os.system("ls -l " + str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    raise SystemExit(main())
