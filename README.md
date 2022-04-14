# rrs_public
Rapid Response System Public code

Predict patient's in-hospital abnormal status for rapid response system 

- Overall system
  + Input: Patient's clinical variables (includes vital signs and laboratory tests) in an observation peroid
  + Output: Patient's abnormal status/ abnormal probability

![overall_dews](https://user-images.githubusercontent.com/35287087/163348381-d9abc484-6138-40c4-9a2d-89e5ce99ebdc.png)

- Window interval processing

![Overall](https://user-images.githubusercontent.com/35287087/163349152-3e5ed442-0777-4b32-992f-0b1fff4fe670.png)

  + Window time D contains the measurement features of patient in n timepoints (in the figure, n = 8)
  + The window time D will slides for every timepoint for each patient
  + With n = 8, for each patient, we should wait 8 hours for the first output
  + After that, the system will predict once every hour



- Install enviroment dependencies via 'requirement.txt'
- Run CNUH_prediction.py for predict the outcome
