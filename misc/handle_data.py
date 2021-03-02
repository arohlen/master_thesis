import h5py

filename = "D:\Shetty_data\data_labels\data_labels.h5"

path = "D:\Shetty_data"

with h5py.File(filename, "r") as f:

    # Get all the labels
    all_sat_LLHATR = list(f["all_sat_LLAHTR"])
    all_uav_LLHATR = list(f["all_uav_LLAHTR"])
    all_uav_xyzHTR = list(f["all_uav_xyzHTR"])
    match_array_40 = list(f["match_array_40"])
    sat300_image_paths = list(f["sat300_image_paths"])
    uav_image_paths = list(f["uav_image_paths"])



print(uav_image_paths[0])
