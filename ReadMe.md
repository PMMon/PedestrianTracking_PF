# Pedestrian Tracking in Video Sequences using Particle Filters

This is the code for the project report: 

P. Mondorf, T. Labourdette-Liaresq and D. Singh: [Pedestrian Tracking in Video Sequences using Particle Filters](EL2320_ProjectReport_PhilippMondorf.pdf), Project Report, Division of Robotics, Perception and Learning, KTH Royal Institute of Technology, 2021. 

In this work, we evaluate and compare three different methods for pedestrian tracking in video sequences
using particle filters. Particle filters have become popular tools in solving visual tracking tasks as they are capable of handling complex non-linear motions and non-Gaussian distributions. For this work, two particle filter
approaches are implemented that use different image features to compare the target pedestrian with respective state estimates. While the first technique is based on HSV color histograms, the
second method makes use of moment invariants. By combining the two particle filter approaches, we are able to create a third
tracking system that benefits from the advantages of both former techniques. We evaluate the introduced methods on a challenging dataset from the BoBoT benchmark for visual object tracking [[1]](#1).


Below we show tracking results of our fused particle filter approach. We further illustrate two error measurements for the estimated state. For details, please see the above mentioned paper.
<br />

![ICM Tracking Results](<ReadMeFiles/GIFs/ParticleFilter_ICM_model.gif>)

Tracking results of the ICM tracking system, evaluated on a dataset from the BoBoT benchmark [[1]](#1)

<br />

If you find this code useful for your research, please cite it as: 

```
@ARTICLE{MondorfPFTracking,
    author = {Philipp Mondorf},
    title = {Pedestrian Tracking in Video Sequences using Particle Filters},
    journal={Technical Report},
    year = {2021}
}
```

## Setup
All code was developed and tested on Windows 10 with Python 3.7.

To run the current code, we recommend to setup a virtual environment: 

```bash
python3 -m venv env                     # Create virtual environment
source env/bin/activate                 # Activate virtual environment
pip install -r requirements.txt         # Install dependencies
# Work for a while
deactivate                              # Deactivate virtual environment
```

## Particle Filters for Pedestrian Tracking

The code in the folder [ParticleFilter](ParticleFilter) implements three different particle filter approaches for pedestrian tracking.
These approaches use different image features to compare the target pedestrian with respective state estimates, i.e. they differ in their observation model: 

- CLR: based on HSV color histograms
- MMT: based on moment invariants
- ICM: combination of both former methods

### Track Pedestrian using default Settings

In order to run the code, navigate to this folder in your command shell and run the following command:

```
python ParticleFilter/PFTracking.py
```

This will track the pedestrian using an implementation of the color-based particle filter approach with N = 100 particles. 
It is possible to configure the tracking process by using command-line flags. A detailed explanation of these flags can be found [here](ReadMeFiles/ARGUMENTS.md).


### Track Pedestrian using command-line flags

Command-line flags can be defined using the `--variable_name` expression in the command shell. To see a list of all available parameters run the following command: 

```
python ParticleFilter/PFTracking.py --help
```

### Example - Determine Number of Particles

As an example, we specify to track the pedestrian using a moment-based particle filter approach with N = 50 particles. 
This can be done by running the following command: 

```
python ParticleFilter/PFTracking.py --OM MMT --N 50
```

### Example - Create Video with Particles

To create a video that shows the underlying particles, simply set the parameter `--video` and the parameter 
`--show_part` to `True`: 

```
python ParticleFilter/PFTracking.py --video True --show_part True
```

A new for the video can be defined using the command-line flag `--vid_name`. An example of such a generated video is shown below: 

<br />

![Color-based Tracking Results](<ReadMeFiles/GIFs/ParticleFilter_CLR_particles.gif>)

Tracking results of the color-based particle filter approach. Particle states are also displayed.

<br />

## Create videos from GT 

It is also possible to generate a video of the dataset from the given frames. For this, run the following command:

```
python GT_Preparation/RunPrep.py
```
 
If you want to visualize bounding boxes, please ensure that the variable `annotate` is set to `True`. Also, make sure that you have specified the input and output paths correctly.

## References
<a id="1">[1]</a>  D. A. Klein. “Bobot - bonn benchmark on tracking”. In: Technical Report (2010). 