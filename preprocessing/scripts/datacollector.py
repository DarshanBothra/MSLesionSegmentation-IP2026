import os
import json

class DataCollector:
    def __init__(self):
        self.DATASET_ROOT = "/home/darshan/MS/data/RAW"
    
    def fetch_isbi_data(self, json_file):

        root = os.path.join(self.DATASET_ROOT, "ISBI2015", "train")
        with open(json_file, "r") as f:
            data = json.load(f)

        data = data["patient_wise"]
        patients = data.keys()
        self.isbi_files = {}
        i = 1
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

                    self.isbi_files[f"S{i}"] = f
                    i+=1

        return self.isbi_files
    
    def test_isbi(self):
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
    
    def fetch_mslegseg_data(self, json_file):
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

                    self.mslegseg_files[f"S{i}"] = f
                    i+=1
                else:
                    print(patient, tp)

        return self.mslegseg_files

    def test_mslegseg(self):
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




if __name__ == "__main__":
    collector = DataCollector()
    
    # isbi
    '''
    isbi_data = collector.fetch_isbi_data("/home/darshan/MS/eda/results/exploratory/isbi_eda.json")
    collector.test_isbi()
    '''

    #mslegseg
    mslegseg_data = collector.fetch_mslegseg_data("/home/darshan/MS/eda/results/exploratory/msleg_eda.json")
    collector.test_mslegseg()


