Data:
  dielectric_spectrums_four.csv
    4 layers, 10-40, step 2's.
    Silver, Silica, Silver, Silica
  double_dielectrics.csv
    4 layers, 10-100, step 6's.
    Silica, TiO2, Silica, TiO2
  test_answer.csv
    Spectrum
  test_small_fixed.csv
    2 layers, 10-40, steps 1. Uses .001 notation
    Silver, Gold
  order_die.csv
     5 layers. silica, ti02, silica, ti02, silica
     Order is 8 on all.
     Note it was made by a hybrid, so it could be inaccurate.
     
  5_layer_tio2_combined.csv
      5 layers. silica, tio2, silica, ti02, silica, 
      Order is 8 on all. Note that tio2 has been corrected to be dispersive.
      The data was generated from a monte-carlo simulation.

Results:
  Dielectric_Corrected_TiO2:
    This is a trained NN for 5 layers, 30-70, monte carlo generated. It has a corrected TiO2.
    Note that it has 4 layers, 75 neurons per layer.
    


Steps to analyze results:
    Run plotLoss.py with the loss file.
    