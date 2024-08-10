import numpy as np
import pandas as pd


def tensor2csv(tensor, filename):
  t = tensor.cpu()
  t_np = t.numpy() #convert to Numpy array
  df = pd.DataFrame(t_np) #convert to a dataframe
  df.to_csv(filename,index=False) #save to file