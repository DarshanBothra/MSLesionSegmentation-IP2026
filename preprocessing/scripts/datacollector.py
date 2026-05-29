"""
This script explores all the directories for MSLegSeg, MICCAI and ISBI2015 dataset using the JSON Files prepared for each dataset during eda. using which fetches all file paths of flair, t1w, t2w samples along with their masks and universally pools them!
"""

import os
import json

class DataCollector:
    def __init__(self):
        self.DATASET_ROOT = "/home/darshan/MS/data/RAW"
    
    def _fetch_isbi_data(self, json_file):

        print("Fetching samples from ISBI2015 dataset...")

        root = os.path.join(self.DATASET_ROOT, "ISBI2015", "train")
        with open(json_file, "r") as f:
            data = json.load(f)

        data = data["patient_wise"]
        patients = data.keys()
        self.isbi_files = {}
        i = 0
        for patient in patients:
            time_points = [x for x in data[patient].keys() if x.startswith('0')]
            for tp in time_points:
                f = {}
                flair_file = os.path.join(root, f"training{patient}", "orig", f"training{patient}_{tp}_flair.nii.gz")
                t1_file = os.path.join(root, f"training{patient}", "orig", f"training{patient}_{tp}_mprage.nii.gz")
                t2_file = os.path.join(root, f"training{patient}", "orig", f"training{patient}_{tp}_t2.nii.gz")
                mask_file = os.path.join(root, f"training{patient}", "masks", f"training{patient}_{tp}_mask1.nii")

                flair_file = flair_file if os.path.exists(flair_file) else None
                t1_file = t1_file if os.path.exists(t1_file) else None
                t2_file = t2_file if os.path.exists(t2_file) else None
                mask_file = mask_file if os.path.exists(mask_file) else None

                if (flair_file and t1_file and t2_file and mask_file):
                    f["flair"] = flair_file
                    f["t1"] = t1_file
                    f["t2"] = t2_file
                    f["mask"] = mask_file
                    f["dataset"] = "isbi"
                    self.isbi_files[f"S{i}"] = f
                    i+=1
                else:
                    print("Skipping ISBI2015: ", patient, tp)

        print(f"{i} Samples loaded\n")
        return self.isbi_files
    
    def _test_isbi(self):
        print("========== ISBI FILES ==========")
        print("Keys: ", self.isbi_files.keys())
        print()
        for key in self.isbi_files:
            print("First key", key)
            print("Files", self.isbi_files[key])
            break
        print()
        print("\tTotal keys: ", len(self.isbi_files.keys()), end = "")
        isbi_flair = 0
        isbi_t1 = 0
        isbi_t2 = 0
        isbi_mask = 0

        for key in self.isbi_files:
            value = self.isbi_files[key]
            for mode in value:
                if mode == "flair":
                    isbi_flair = isbi_flair + 1 if value[mode] is not None else isbi_flair
                elif mode == "t1":
                    isbi_t1 = isbi_t1 + 1 if value[mode] is not None else isbi_t1
                elif mode == "t2":
                    isbi_t2 = isbi_t2 + 1 if value[mode] is not None else isbi_t2
                elif mode == "mask":
                    isbi_mask = isbi_mask + 1 if value[mode] is not None else isbi_mask

        print(f"""
        Total flair: {isbi_flair}
        Total t1: {isbi_t1}
        Total t2: {isbi_t2}
        Total mask: {isbi_mask}
        """)
    
    def _fetch_mslegseg_data(self, json_file):
        print("Fetching samples from MSLegSeg dataset...")
        data_root = os.path.join(self.DATASET_ROOT, "MSLegSeg", "MSLegSeg_RAW")
        mask_root = os.path.join(self.DATASET_ROOT, "MSLegSeg", "MSLegSeg Dataset")

        with open(json_file, "r") as f:
            data = json.load(f)
        
        data = data["patient_wise"]
        patients = [x for x in data.keys() if x.startswith("P")]
        self.mslegseg_files = {}
        i = 1
        for patient in patients:
            time_points = [x for x in data[patient] if x.startswith("T")]
            for tp in time_points:
                f = {}
                flair_file = os.path.join(data_root, patient, tp , f"{patient}_{tp}_FLAIR.nii.gz")
                t1_file = os.path.join(data_root, patient, tp , f"{patient}_{tp}_T1.nii.gz")
                t2_file = os.path.join(data_root, patient, tp , f"{patient}_{tp}_T2.nii.gz")

                flair_file = flair_file if os.path.exists(flair_file) else None
                t1_file = t1_file if os.path.exists(t1_file) else None
                t2_file = t2_file if os.path.exists(t2_file) else None

                # mask could be in test or train
                
                # mask is in train

                mask_file = os.path.join(mask_root, "train", patient, tp, f"{patient}_{tp}_MASK.nii.gz")
                
                # check if mask file is in test
                mask_file = mask_file if os.path.exists(mask_file) else os.path.join(mask_root, "test", patient, f"{patient}_MASK.nii.gz")

                # set none if mask file does not exist
                mask_file = mask_file if os.path.exists(mask_file) else None

                if (flair_file and t1_file and t2_file and mask_file):
                    f["flair"] = flair_file
                    f["t1"] = t1_file
                    f["t2"] = t2_file
                    f["mask"] = mask_file
                    f["dataset"] = "mslegseg"

                    self.mslegseg_files[f"S{i}"] = f
                    i+=1
                else:
                    print("Skipping MSLegSeg: ", patient, tp)
        print(f"{i} samples loaded\n")
        return self.mslegseg_files

    def _test_mslegseg(self):
        print("========== MSLEGSEG FILES ==========")
        print("Keys: ", self.mslegseg_files.keys())
        print()
        for key in self.mslegseg_files:
            print("First key", key)
            print("Files", self.mslegseg_files[key])
            break
        print()
        print("\tTotal keys: ", len(self.mslegseg_files.keys()), end = "")
        isbi_flair = 0
        isbi_t1 = 0
        isbi_t2 = 0
        isbi_mask = 0
        
        for key in self.mslegseg_files:
            value = self.mslegseg_files[key]
            for mode in value:
                if mode == "flair":
                    isbi_flair = isbi_flair + 1 if value[mode] is not None else isbi_flair
                elif mode == "t1":
                    isbi_t1 = isbi_t1 + 1 if value[mode] is not None else isbi_t1
                elif mode == "t2":
                    isbi_t2 = isbi_t2 + 1 if value[mode] is not None else isbi_t2
                elif mode == "mask":
                    isbi_mask = isbi_mask + 1 if value[mode] is not None else isbi_mask

        print(f"""
        Total flair: {isbi_flair}
        Total t1: {isbi_t1}
        Total t2: {isbi_t2}
        Total mask: {isbi_mask}
        """)

    def _fetch_miccai_data(self, json_file):
        print("Fetching samples from MICCAI2016 dataset...")
        root = os.path.join(self.DATASET_ROOT, "MICCAI2016")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        training_centers = [x for x in data["training"].keys() if x.startswith("Center")]
        testing_centers = [x for x in data["testing"].keys() if x.startswith("Center")]

        i = 1
        self.miccai_files = {}

        for center in training_centers:
            patients = [x for x in data["training"][center].keys() if x.startswith("Patient")]
            for patient in patients:
                f = {}
                flair_file = os.path.join(root, "Training", center , patient, "Raw_Data", "FLAIR.nii.gz")
                t1_file = os.path.join(root, "Training", center , patient, "Raw_Data", "T1.nii.gz")
                t2_file = os.path.join(root, "Training", center , patient, "Raw_Data", "T2.nii.gz")

                flair_file = flair_file if os.path.exists(flair_file) else None
                t1_file = t1_file if os.path.exists(t1_file) else None
                t2_file = t2_file if os.path.exists(t2_file) else None

                mask_file = os.path.join(root, "Training", center , patient, "Masks", "ManualSegmentation_1.nii.gz")

                # set none if mask file does not exist
                mask_file = mask_file if os.path.exists(mask_file) else None

                if (flair_file and t1_file and t2_file and mask_file):
                    f["flair"] = flair_file
                    f["t1"] = t1_file
                    f["t2"] = t2_file
                    f["mask"] = mask_file
                    f["dataset"] = "miccai"

                    self.miccai_files[f"S{i}"] = f
                    i+=1
                else:
                    print("Skipping MICCAI 2016: ", center, patient)
        
        for center in testing_centers:
            patients = [x for x in data["testing"][center].keys() if x.startswith("Patient")]
            for patient in patients:
                f = {}
                flair_file = os.path.join(root, "Testing", center , patient, "Raw_Data", "FLAIR.nii.gz")
                t1_file = os.path.join(root, "Testing", center , patient, "Raw_Data", "T1.nii.gz")
                t2_file = os.path.join(root, "Testing", center , patient, "Raw_Data", "T2.nii.gz")

                flair_file = flair_file if os.path.exists(flair_file) else None
                t1_file = t1_file if os.path.exists(t1_file) else None
                t2_file = t2_file if os.path.exists(t2_file) else None

                mask_file = os.path.join(root, "Testing", center , patient, "Masks", "ManualSegmentation_1.nii.gz")

                # set none if mask file does not exist
                mask_file = mask_file if os.path.exists(mask_file) else None

                if (flair_file and t1_file and t2_file and mask_file):
                    f["flair"] = flair_file
                    f["t1"] = t1_file
                    f["t2"] = t2_file
                    f["mask"] = mask_file
                    f["dataset"] = "miccai"

                    self.miccai_files[f"S{i}"] = f
                    i+=1
                else:
                    print(center, patient)
        print(f"{i} samples loaded\n")
        return self.miccai_files
    
    def _test_miccai(self):
        print("========== MICCAI FILES ==========")
        print("Keys: ", self.miccai_files.keys())
        print()
        for key in self.miccai_files:
            print("First key", key)
            print("Files", self.miccai_files[key])
            break
        print()
        print("\tTotal keys: ", len(self.miccai_files.keys()), end = "")
        isbi_flair = 0
        isbi_t1 = 0
        isbi_t2 = 0
        isbi_mask = 0
        
        for key in self.miccai_files:
            value = self.miccai_files[key]
            for mode in value:
                if mode == "flair":
                    isbi_flair = isbi_flair + 1 if value[mode] is not None else isbi_flair
                elif mode == "t1":
                    isbi_t1 = isbi_t1 + 1 if value[mode] is not None else isbi_t1
                elif mode == "t2":
                    isbi_t2 = isbi_t2 + 1 if value[mode] is not None else isbi_t2
                elif mode == "mask":
                    isbi_mask = isbi_mask + 1 if value[mode] is not None else isbi_mask

        print(f"""
        Total flair: {isbi_flair}
        Total t1: {isbi_t1}
        Total t2: {isbi_t2}
        Total mask: {isbi_mask}
        """)

    def pool_data(self):
            
        isbi_data = self._fetch_isbi_data("/home/darshan/MS/eda/results/exploratory/isbi_eda.json")
        mslegseg_data = self._fetch_mslegseg_data("/home/darshan/MS/eda/results/exploratory/msleg_eda.json")
        miccai_data = self._fetch_miccai_data("/home/darshan/MS/eda/results/exploratory/miccai2016_eda.json")

        # universally combine all data
        self.data = {}
        i = 1
        for dataset in [isbi_data, mslegseg_data, miccai_data]:
            for key in dataset:
                self.data[f"S{i}"] = dataset[key]
                i+=1

        return self.data


if __name__ == "__main__":
    collector = DataCollector()
    
    # # isbi
    # isbi_data = collector._fetch_isbi_data("/home/darshan/MS/eda/results/exploratory/isbi_eda.json")
    # collector.test_isbi()

    # #mslegseg
    # mslegseg_data = collector._fetch_mslegseg_data("/home/darshan/MS/eda/results/exploratory/msleg_eda.json")
    # collector._test_mslegseg()
    # collector.test_mslegseg()

    # #miccai
    # miccai_data = collector._fetch_miccai_data("/home/darshan/MS/eda/results/exploratory/miccai2016_eda.json")
    # collector.test_miccai()

    data = collector.pool_data()
    flair_count = 0
    t1_count = 0
    t2_count = 0
    mask_count = 0
    print("Keys: ", data.keys())
    print("Total keys: ", len(data.keys()))

    for key in data.keys():
        if data[key]["flair"] is not None:
            flair_count +=1
        if data[key]["t1"] is not None:
            t1_count +=1
        if data[key]["t2"] is not None:
            t2_count +=1
        if data[key]["mask"] is not None:
            mask_count +=1

    print(f"Total flair: {flair_count}")
    print(f"Total t1: {t1_count}")
    print(f"Total t2: {t2_count}")
    print(f"Total mask: {mask_count}")

    OUTPUT_FILE = "/home/darshan/MS/preprocessing/samples.json"
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Sample paths saved to {OUTPUT_FILE}")

