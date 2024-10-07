import pandas as pd
import os

import fazy.prva_faza as prva_faza
os.makedirs('plots', exist_ok=True)

def main():
    # Load the data
    df_conn = pd.read_csv('024/connections.csv', sep='\t')
    df_dev = pd.read_csv('024/devices.csv', sep='\t')
    df_proc = pd.read_csv('024/processes.csv', sep='\t')
    df_prof = pd.read_csv('024/profiles.csv', on_bad_lines='skip', sep='\t')

    # Check if the data is loaded
    if df_conn.empty or df_dev.empty or df_proc.empty or df_prof.empty:
        print('Data is empty!')
        return -1
    else:
        print('Data loaded successfully!')
        print('________________________________________')

        datasets = {
        'Connections': df_conn,
        'Devices': df_dev,
        'Processes': df_proc,
        'Profiles': df_prof
    }
        i=0
        attributes = [['c.android.youtube','c.dogalize','c.android.gm','c.katana','c.android.chrome','c.updateassist','c.android.vending'],[],['p.android.chrome','p.android.settings','p.android.documentsui','p.system','p.android.externalstorage','p.android.packageinstaller','p.android.gm','p.dogalize','p.olauncher','p.process.gapps'],[]]


        for name, data in datasets.items():
            # 1.1  task
            prva_faza.A1b([(name, data)])
            prva_faza.B1b([(name, data)],attributes[i])
            prva_faza.C1b([(name, data)])
            i += 1

            if name == 'Connections':
                middle_corr_pairs = [('c.dogalize', 'c.android.gm'), ('c.android.gm', 'c.katana')]
                minimal_corr_pairs = [
                    ('mwra', 'c.dogalize'), ('mwra', 'c.android.gm'), ('mwra', 'c.katana'),
                    ('mwra', 'c.android.chrome'), ('c.dogalize', 'c.katana'), ('c.dogalize', 'c.android.chrome')
                ]

                for pair in middle_corr_pairs:
                    prva_faza.D1b(pair[0], pair[1], [(name, data)])
                for pair in minimal_corr_pairs:
                    prva_faza.D1b(pair[0], pair[1], [(name, data)])
            elif name == 'Devices':
                # we dont have big correlation
                pass
            elif name == 'Processes':
                big_corr_pairs = [('p.system', 'p.android.gm')]
                middle_corr_pairs = [
                    ('mwra', 'p.android.settings'), ('mwra', 'p.system'), ('mwra', 'p.android.gm'),
                    ('p.system', 'p.android.externalstorage')
                ]
                minimal_corr_pairs = [
                    ('mwra', 'p.android.externalstorage'), ('mwra', 'p.android.packageinstaller'),
                    ('p.android.settings', 'p.system'), ('p.android.settings', 'p.android.gm'),
                    ('p.android.documentsui', 'p.android.externalstorage'),
                    ('p.android.externalstorage', 'p.android.gm'), ('p.android.packageinstaller', 'p.android.gm')
                ]

                for pair in big_corr_pairs:
                    prva_faza.D1b(pair[0], pair[1], [(name, data)])
                for pair in middle_corr_pairs:
                    prva_faza.D1b(pair[0], pair[1], [(name, data)])
                for pair in minimal_corr_pairs:
                    prva_faza.D1b(pair[0], pair[1], [(name, data)])
            elif name == 'Profiles':
                # we dont have big correlation
                pass
            else:
                print('Error: Unknown dataset!')
                return -1

            # 1.2 task
            prva_faza.clean_all_data(name)

        prva_faza.E1b()

        print("##################################################")
        print("##################################################\n")

    return 1

if __name__ == '__main__':
    main()