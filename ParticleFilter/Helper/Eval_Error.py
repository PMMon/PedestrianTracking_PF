# ==Imports==
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# ========================
#  Evaluation of errors
# ========================

class Eval:
    def __init__(self, args, output_path):
        self.args = args

        self.path_excel = os.path.join(output_path, "excel_sheets")

        if not os.path.exists(self.path_excel):
            os.makedirs(self.path_excel)


    def evalerror(self, error, OM, filename="ParticleFilter.xlsx"):
        """
        Write errors to .xlsx-file
        :param error: Dictionary with types of errors as key and array of error values as value
        :param OM: Type of observation model
        :param filename: Name of .xlsx-file
        """

        df_dict = {"Filter": "Particle Filter", "OM": OM, "N": self.args.N, "alpha": self.args.alpha, "exp_nr": self.args.exp_nr, "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
        for key, val in error.items():
            df_dict["Min. error " + key] = np.min(val)
            df_dict["Max. error " + key] = np.max(val)
            df_dict["Avg. error " + key] = np.mean(val)
            df_dict["Avg. first third " + key] = np.mean(val[0:round(len(val)*1/3)])
            df_dict["Avg. second third " + key] = np.mean(val[round(len(val) * 1 / 3) : round(len(val) * 2 / 3)])
            df_dict["Avg. three third " + key] = np.mean(val[round(len(val) * 2 / 3) : len(val)])

        file_spec = os.path.join(self.path_excel, filename)
        if not os.path.exists(file_spec):
            df = pd.DataFrame(df_dict, index=[0])

        else:
            df = pd.read_excel(file_spec, engine="openpyxl")
            df_new = pd.DataFrame(df_dict, index=[0])
            df = df.append(df_new, ignore_index=True, sort=True)

        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        writer = pd.ExcelWriter(file_spec, engine="openpyxl")
        df.to_excel(writer)
        writer.save()


