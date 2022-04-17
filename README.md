# rrs_public
Rapid Response System Public code

Predict patient's in-hospital abnormal status for rapid response system 

# Overall system
  + Input: Patient's clinical variables (includes vital signs and laboratory tests) in an observation peroid
  + Output: Patient's abnormal status/ abnormal probability

![overall_dews](https://user-images.githubusercontent.com/35287087/163348381-d9abc484-6138-40c4-9a2d-89e5ce99ebdc.png)

# Window interval processing
  + Window time D contains the measurement features of patient in n timepoints (in the figure, n = 8)
  + The window time D will slides for every timepoint for each patient
  + With n = 8, for each patient, we should wait 8 hours for the first output
  + After that, the system will predict once every hour

![Overall](https://user-images.githubusercontent.com/35287087/163349152-3e5ed442-0777-4b32-992f-0b1fff4fe670.png)

- Proposed model: Temporal Variational Autoencoder (TVAE)

![tvae](https://user-images.githubusercontent.com/35287087/163350134-c0ebd45c-2bfe-42a7-b5f4-a38b1f265b4b.png)


# Running:
  + Install enviroment dependencies via 'requirement.txt'
  + Install torch: conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  + Configure the file path on the functions
  + Run 'CNUH_prediction.py' for predict the outcome
