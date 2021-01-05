# ==Imports==
import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Helper.Bool_Flag import bool_flag

# ================
#  Plot error
# ================

class PlotError:
    def __init__(self, args, input_path, output_path):
        self.args = args

        if not os.path.exists(input_path):
            raise FileNotFoundError("Path %s does not exist!", input_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.input_path = input_path
        self.output_path = output_path


    def plot_error(self):
        """
        Plot error of type self.args.error
        """
        tracking_errors = np.array([])

        print("read in errors...")
        for i in range(self.args.number):
            filename = "Error_PF_" + str(self.args.OM) + "_" + str(self.args.error) + "_" + str(i+1) + ".txt"
            if i == 0:
                try:
                    tracking_errors = np.append(tracking_errors, np.loadtxt(os.path.join(self.input_path, filename)))
                except:
                    print("File %s does not exist! Skip this file..." % os.path.join(self.input_path, filename))
            else:
                try:
                    tracking_errors = np.vstack((tracking_errors, np.loadtxt(os.path.join(self.input_path, filename))))
                except:
                    print("File %s does not exist! Skip this file..." % os.path.join(self.input_path, filename))

        print("array of errors created.\ncalculate mean and std deviation")

        mean_array = np.mean(tracking_errors, axis = 0)
        std_dev_array = np.std(tracking_errors, axis = 0)

        # Calculate evaluation for parts
        print("Error type: " + str(self.args.error))

        final_mean = np.mean(mean_array)
        final_std_dev = np.mean(std_dev_array)

        print("Final mean: " + str(final_mean))
        print("Final Std_dev: " + str(final_std_dev))

        # Thirds
        first_third_mean = np.mean(mean_array[0:round(len(mean_array) * 1 / 3)])
        first_third_std_dev = np.mean(std_dev_array[0:round(len(mean_array) * 1 / 3)])
        print("First third mean: " + str(first_third_mean))
        print("First third Std_dev: " + str(first_third_std_dev))

        second_third_mean = np.mean(mean_array[round(len(mean_array) * 1 / 3): round(len(mean_array) * 2 / 3)])
        second_third_std_dev = np.mean(std_dev_array[round(len(mean_array) * 1 / 3): round(len(mean_array) * 2 / 3)])
        print("Second third mean: " + str(second_third_mean))
        print("Second third Std_dev: " + str(second_third_std_dev))

        last_third_mean = np.mean(mean_array[round(len(mean_array) * 2 / 3): len(mean_array)])
        last_third_std_dev = np.mean(std_dev_array[round(len(mean_array) * 2 / 3): len(mean_array)])
        print("Last third mean: " + str(last_third_mean))
        print("Last third Std_dev: " + str(last_third_std_dev))

        # Create Plot
        if self.args.error == "overlap_area":
            print("plot error...")

            plt.rc('font', family='serif', serif='Times')
            plt.rc('text', usetex=True)
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            plt.rc('axes', labelsize=14)


            # width as measured in inkscape
            width = 1.5*3.487
            height = width / 1.618

            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

            x = np.arange(mean_array.shape[0]) + 1
            ax.xaxis.set_ticks(np.arange(0, mean_array.shape[0] + 1, 100), minor=True)
            plt.grid(linestyle='--', color='silver', which='both')
            plt.plot(x, mean_array, 'b-', linewidth=1.0)
            plt.fill_between(x, mean_array-std_dev_array, mean_array+std_dev_array, facecolor='darkgray', edgecolor='darkgray')

            # Axis
            ax.set_ylabel("Overlap ($\mu$ for 10 runs)")
            ax.set_xlabel('Frames')
            ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))

            #ax.tick_params(axis='x', which='minor', bottom=False)

            # Text box
            textstr = '; '.join((r'$\mu=%.2f$' % (final_mean,),
                r'$\sigma=%.2f$' % (final_std_dev,)))

            props = dict(boxstyle='round', facecolor='none')

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.06, textstr, fontsize=14, transform=ax.transAxes, verticalalignment='bottom', bbox=props)

            fig.set_size_inches(width, height)

            plot_name = self.args.error + '_' + self.args.OM + '.jpg'
            fig.savefig(os.path.join(self.output_path, plot_name), dpi=200)

            print("figure created.")



if __name__ == "__main__":
    # Paths
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'output', 'eval', 'error_files'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'output', 'eval', 'error_plots'))

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Error Plotting for Pedestrian Tracking")

    # General configs for Plotting
    parser.add_argument("--error", default="overlap_area", type=str, help="Specify which error type should be computed. Choose either euclidean, area or overlap_area")
    parser.add_argument("--number", default=10, type=int, help="Specify number of experiments that should be evaluated")

    # Configs for Tracking
    parser.add_argument("--OM", default="CLR", type=str, help="Specify which observation model to use. Choose either CLR, MMT or CM")


    # Get arguments
    args = parser.parse_args()

    # Run Tracker
    tracker = PlotError(args, input_path, output_path)
    tracker.plot_error()