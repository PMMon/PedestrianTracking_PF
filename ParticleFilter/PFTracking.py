# ==Imports==
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import argparse
import cv2
import numpy as np
import imutils
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ParticleFilter.Helper.Bool_Flag import bool_flag
from ParticleFilter.Helper.Eval_Error import Eval
from ParticleFilter.ObjectDetection.PedDetector import PedestrianDetector
from ParticleFilter.Filters.ParticleFilter import PF

# ===============================================================
#  Tracking Pedestrians in video sequences with Particle Filter
# ===============================================================

class PedestrianTracker:
    def __init__(self, args, input_path, output_path):
        self.args = args
        self.PF = PF(args)

        if not os.path.exists(input_path):
            raise FileNotFoundError("Path %s does not exist!", input_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.input_path = input_path
        self.output_path = output_path

        # Paths for videos and evaluations
        self.video_path = os.path.join(output_path, "videos")
        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)

        self.video_images = os.path.join(output_path, "vidimages")
        if not os.path.exists(self.video_images):
            os.makedirs(self.video_images)

        if self.args.vid_name != "":
            self.video_name = "ParticleFilter_" + self.args.OM + "_" + self.args.vid_name + ".mp4"
        else:
            self.video_name = "ParticleFilter_" + self.args.OM + ".mp4"

        self.evaluation_path = os.path.join(output_path, "eval")
        if not os.path.exists(self.evaluation_path):
            os.makedirs(self.evaluation_path)

        # Define variance for motion model
        self.var_c = np.array([20, 30, 10, 10, 40, 40, 0.1])
        self.var_m = np.array([20, 30, 10, 10, 40, 40, 0.1])

        # Define bounds for motion model
        self.bounds_c = np.array([25, 25, 30, 100, 60, 150, 0.05])
        self.bounds_m = np.array([25, 25, 30, 100, 60, 150, 0.05])


    def trackpedestrian(self, gt_filename = "gt.txt"):
        """
        Tracking pedestrians using Particle Filter. Specific configurations are given by the input arguments
        :param gt_filename: Name of file with information about ground truth
        """
        imgs = []
        states_clr = np.array([])
        states_mmt = np.array([])
        states_fusion = np.array([])

        processed_frames = 0
        initial_alpha = self.args.alpha
        initial_var_c = self.args.sigma_c
        initial_var_m = self.args.sigma_m

        # Get GT positions from file
        color_GT = (255, 0, 0)
        bboxes = open(os.path.join(self.input_path, gt_filename), "r")
        positions = bboxes.readlines()
        positions = [p.replace("\n", "").split(",") for p in positions]


        # ============================= Initialization =============================
        print("Start Tracking...")

        # Detection of target, using classical CV
        detector = PedestrianDetector(self.args.detect, self.args.colorbased, self.args.color)

        for i, file in enumerate(os.listdir(self.input_path)):
            if file.endswith(".jpg") or file.endswith(".png"):
                frame = cv2.imread(os.path.join(self.input_path, file), -1)
                regions, confidence = detector.detectped(frame)
                processed_frames += 1

                if not regions:
                    print("Unable to detect pedestrian! Process next frame...")
                    continue
                elif len(regions) >= 2:
                    print("More than one possible target detected! Process next frame...")
                    continue
                elif confidence[0] <= 2.0:
                    print("Confidence of target detection not high enough! Process next frame...")
                    continue
                else:
                    break

        # Get ground truth for current frame if available
        try:
            GT_x = round(float(positions[processed_frames][0]))
            GT_y = round(float(positions[processed_frames][1]))
            GT_x_w = round(float(positions[processed_frames][2]))
            GT_y_h = round(float(positions[processed_frames][3]))
            GT_w = abs(GT_x_w - GT_x)
            GT_h = abs(GT_y_h - GT_y)
            GT = np.array([[GT_x], [GT_y], [0], [0], [GT_w], [GT_h], [1]])
            gt_update = True
        except:
            gt_update = False

        # Initialize state vector for target
        vx_initial = 0
        vy_initial = 0
        width = regions[0][2]
        height = regions[0][3]
        target = np.array([[int(regions[0][0])], [int(regions[0][1])], [vx_initial], [vy_initial], [width], [height], [1]])

        # Initialize particles
        particles_m, weights_m = self.PF.initparticles(frame.shape)
        particles_c, weights_c = self.PF.initparticles(frame.shape)

        # == Initialize feature vectors respective to observation model ==

        # Observation model based on invariant moments
        if self.args.OM.upper() == "MMT" or self.args.OM.upper() == "CM":
            err_m_euclidean = np.array([])
            err_m_area = np.array([])
            err_m_overlap_area = np.array([])

            # Define observation model
            obsmodel_moments = self.PF.observationmodel(self.args.OM.upper(), 1)

            # Get state vectors of particles and target
            t_moments = obsmodel_moments.initialization(target, frame)
            p_moments = obsmodel_moments.initialization(particles_m, frame)

            # Calculate weights based on moment vectors
            weights_m = obsmodel_moments.likelihood(t_moments, p_moments, self.args.mu_m, self.args.sigma_m)

            # Estimate weighted mean of all particles as final estimation
            est_m = np.dot(particles_m, weights_m).reshape(7,1)
            states_mmt = np.append(states_mmt, est_m).reshape(1,7)

            # Compute errors if ground truth is available
            if gt_update:
                err_m_euclidean = np.append(err_m_euclidean, self.calculate_error(GT, est_m, "euclidean"))
                err_m_area_value, err_m_overlap_area_value = self.calculate_error(GT, est_m, "area")

                err_m_area = np.append(err_m_area, err_m_area_value)
                err_m_overlap_area = np.append(err_m_overlap_area, err_m_overlap_area_value)

                if self.args.OM != "CM" and self.args.videoloss:
                    if self.args.error == "euclidean":
                        self.ploterror({"euclidean": err_m_euclidean}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "area":
                        self.ploterror({"area": err_m_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "overlap_area":
                        self.ploterror({"overlap_area": err_m_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "both":
                        self.ploterror({"euclidean": err_m_euclidean, "overlap_area": err_m_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    else:
                        raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

            # Show bounding boxes of particles
            if self.args.show_part:
                for p in range(particles_m.shape[1]):
                    cv2.rectangle(frame, (int(particles_m[0, p]), int(particles_m[1, p])), (int(particles_m[0, p] + particles_m[4, p]), int(particles_m[1, p] + particles_m[5, p])), (0, 128, 255), 1)

            # Show weighted average estimate
            if self.args.OM != "CM":
                cv2.rectangle(frame, (int(est_m[0]), int(est_m[1])), (int(est_m[0] + est_m[4]), int(est_m[1] + est_m[5])), (0, 255, 0), 2)

        # Observation model based on color histogram
        if self.args.OM.upper() == "CLR" or self.args.OM.upper() == "CM":
            err_c_euclidean = np.array([])
            err_c_area = np.array([])
            err_c_overlap_area = np.array([])

            # Define observation model
            obsmodel_colors = self.PF.observationmodel(self.args.OM.upper(), 2)

            # Get state vectors of particles and target
            t_colors = obsmodel_colors.initialization(target, frame)
            p_colors = obsmodel_colors.initialization(particles_c, frame)

            # Calculate weights based on color vectors
            weights_c = obsmodel_colors.likelihood(t_colors, p_colors, self.args.mu_c, self.args.sigma_c)

            # Estimate weighted mean of all particles as final estimation
            est_c = np.dot(particles_c, weights_c).reshape(7,1)
            states_clr = np.append(states_clr, est_c).reshape(1,7)

            # Compute error if ground truth is available
            if gt_update:
                err_c_euclidean = np.append(err_c_euclidean, self.calculate_error(GT, est_c, "euclidean"))
                err_c_area_value, err_c_overlap_area_value = self.calculate_error(GT, est_c, "area")
                err_c_area = np.append(err_c_area, err_c_area_value)
                err_c_overlap_area = np.append(err_c_overlap_area, err_c_overlap_area_value)
                if self.args.OM != "CM" and self.args.videoloss:
                    if self.args.error == "euclidean":
                        self.ploterror({"euclidean": err_c_euclidean}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "area":
                        self.ploterror({"area": err_c_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "overlap_area":
                        self.ploterror({"overlap_area": err_c_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "both":
                        self.ploterror({"euclidean": err_c_euclidean, "overlap_area": err_c_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    else:
                        raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

            # Show bounding boxes of particles
            if self.args.show_part:
                for p in range(particles_c.shape[1]):
                    cv2.rectangle(frame, (int(particles_c[0, p]), int(particles_c[1, p])), (int(particles_c[0, p] + particles_c[4, p]), int(particles_c[1, p] + particles_c[5, p])), (0, 128, 255), 1)

            # Show weighted average estimate
            if self.args.OM != "CM":
                cv2.rectangle(frame, (int(est_c[0]), int(est_c[1])), (int(est_c[0] + est_c[4]), int(est_c[1] + est_c[5])), (0, 255, 0), 2)

        # Fusion of both observation models - integrated method
        if self.args.OM == "CM":
            err_euclidean = np.array([])
            err_area = np.array([])
            err_overlap_area = np.array([])

            dclr = self.calculate_error(target, est_c, "euclidean")
            dmmt = self.calculate_error(target, est_m, "euclidean")

            Wclr, Wmmt = self.fuse_states(self.args.beta, dclr, dmmt)

            est_f = Wclr * est_c + Wmmt * est_m
            states_fusion = np.append(states_fusion, est_f).reshape(1,7)

            # Show weighted average estimate
            cv2.rectangle(frame, (int(est_f[0]), int(est_f[1])), (int(est_f[0] + est_f[4]), int(est_f[1] + est_f[5])),(0, 255, 0), 2)

            # Compute errors if ground truth is available
            if gt_update:
                err_euclidean = np.append(err_euclidean, self.calculate_error(GT, est_f, "euclidean"))
                err_area_value, err_overlap_area_value = self.calculate_error(GT, est_f, "area")
                err_area = np.append(err_area, err_area_value)
                err_overlap_area = np.append(err_overlap_area, err_overlap_area_value)
                if self.args.videoloss:
                    if self.args.error == "euclidean":
                        self.ploterror({"euclidean": err_euclidean}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "area":
                        self.ploterror({"area": err_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "overlap_area":
                        self.ploterror({"overlap_area": err_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    elif self.args.error == "both":
                        self.ploterror({"euclidean": err_euclidean, "overlap_area": err_overlap_area}, self.args.OM, processed_frames, self.video_images)
                    else:
                        raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

        # Show GT in frame
        if self.args.show_GT:
            if gt_update:
                cv2.rectangle(frame, (GT_x, GT_y), (GT_x_w, GT_y_h), color_GT, 2)

        # Show DT in frame
        if self.args.show_DT:
                cv2.rectangle(frame, (int(target[0]), int(target[1])),(int(target[0] + target[4]), int(target[1] + target[5])), (0, 0, 255), 2)

        # Showing the output Image
        if self.args.show_frames:
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        imgs.append(frame)


        # ================= Estimation, Update and Resampling ===================
        print("Iterate over frames...")
        current_frame = 1

        # Iterate over frames and track object
        for i, file in enumerate(os.listdir(self.input_path)):
            if file.endswith(".jpg") or file.endswith(".png"):
                if current_frame <= processed_frames:
                    current_frame += 1
                    continue

                if i % 100 == 0:
                    print("Frames processed: %d/%d" % (i, len(os.listdir(self.input_path))))

                # Get frame
                frame = cv2.imread(os.path.join(self.input_path, file), -1)

                # Get ground truth for frame if available
                try:
                    GT_x = round(float(positions[current_frame][0]))
                    GT_y = round(float(positions[current_frame][1]))
                    GT_x_w = round(float(positions[current_frame][2]))
                    GT_y_h = round(float(positions[current_frame][3]))
                    GT_w = abs(GT_x_w - GT_x)
                    GT_h = abs(GT_y_h - GT_y)
                    GT = np.array([[GT_x], [GT_y], [0], [0], [GT_w], [GT_h], [1]])
                    gt_update = True
                except:
                    gt_update = False

                # Resample and estimate
                if self.args.OM.upper() == "MMT" or self.args.OM.upper() == "CM":
                    particles_m, weights_m = self.PF.resampling.resample(particles_m, weights_m)
                    particles_m = self.PF.motion.propagate(particles_m, self.args.dt, self.var_m, frame.shape, self.bounds_m)
                if self.args.OM.upper() == "CLR" or self.args.OM.upper() == "CM":
                    particles_c, weights_c = self.PF.resampling.resample(particles_c, weights_c)
                    particles_c = self.PF.motion.propagate(particles_c, self.args.dt, self.var_c, frame.shape, self.bounds_c)

                # Update target state if object detection is good enough
                regions, confidence = detector.detectped(frame)

                if len(regions) == 1 and confidence[0] > 1.8:
                    width = regions[0][2]
                    height = regions[0][3]
                    target = np.array([[int(regions[0][0])], [int(regions[0][1])], [1/self.args.dt*(regions[0][0] - target[0,0])], [1/self.args.dt*(regions[0][1] - target[1,0])], [width], [height], [(width*height)/(target[4, 0]*target[5,0])]])
                    self.args.alpha = initial_alpha
                    if confidence[0] > 3.5:
                        self.args.sigma_c = 0.01
                        self.args.sigma_m = 0.01
                    else:
                        self.args.sigma_c = initial_var_c
                        self.args.sigma_m = initial_var_m
                else:
                    # don't trust old target anymore
                    self.args.alpha = 0.5
                    self.args.sigma_c = 1000
                    self.args.sigma_m = 1000

                # == Update Step - include observations ==

                # Observation model based on invariant moments
                if self.args.OM.upper() == "MMT" or self.args.OM.upper() == "CM":
                    if self.args.balance:
                        t_moments = (1-self.args.alpha)*obsmodel_moments.initialization(target, frame) + self.args.alpha*obsmodel_moments.initialization(est_m, frame)
                    else:
                        t_moments = obsmodel_moments.initialization(target, frame)

                    p_moments = obsmodel_moments.initialization(particles_m, frame)

                    # Calculate weights based on moment vectors
                    weights_m = obsmodel_moments.likelihood(t_moments, p_moments, self.args.mu_m, self.args.sigma_m)

                    # Estimate weighted mean of all particles as final estimation
                    if len(weights_m[np.where(weights_m < self.args.thresh_m)]) != len(weights_m):
                        weights_m[np.where(weights_m < self.args.thresh_m)] = 0
                    if np.sum(weights_m, axis=0) != 0:
                        weights_m = weights_m/np.sum(weights_m)
                    else:
                        weights_m = 1/len(weights_m) * np.ones(weights_m.shape)

                    # Get weighted average as final estimation
                    est_m = np.dot(particles_m, weights_m).reshape(7,1)
                    states_mmt = np.append(states_mmt, est_m.reshape(1, 7), axis=0)

                    # Compute errors if ground truth is available
                    if gt_update:
                        err_m_euclidean = np.append(err_m_euclidean, self.calculate_error(GT, est_m, "euclidean"))
                        err_m_area_value, err_m_overlap_area_value = self.calculate_error(GT, est_m, "area")

                        err_m_area = np.append(err_m_area, err_m_area_value)
                        err_m_overlap_area = np.append(err_m_overlap_area, err_m_overlap_area_value)

                        if self.args.OM != "CM" and self.args.videoloss:
                            if self.args.error == "euclidean":
                                self.ploterror({"euclidean": err_m_euclidean}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "area":
                                self.ploterror({"area": err_m_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "overlap_area":
                                self.ploterror({"overlap_area": err_m_overlap_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "both":
                                self.ploterror({"euclidean": err_m_euclidean, "overlap_area": err_m_overlap_area}, self.args.OM, current_frame, self.video_images)
                            else:
                                raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

                    # Show bounding boxes of particles
                    if self.args.show_part:
                        for p in range(particles_m.shape[1]):
                            cv2.rectangle(frame, (int(particles_m[0, p]), int(particles_m[1, p])), (int(particles_m[0, p] + particles_m[4, p]), int(particles_m[1, p] + particles_m[5, p])), (0, 128, 255), 1)

                    # Show weighted average estimate
                    if self.args.OM != "CM":
                        cv2.rectangle(frame, (int(est_m[0]), int(est_m[1])), (int(est_m[0] + est_m[4]), int(est_m[1] + est_m[5])), (0, 255, 0), 2)


                # Observation model based on color histogram
                if self.args.OM.upper() == "CLR" or self.args.OM.upper() == "CM":
                    if self.args.balance:
                        t_colors = (1-self.args.alpha)*obsmodel_colors.initialization(target, frame) + self.args.alpha*obsmodel_colors.initialization(est_c, frame)
                    else:
                        t_colors = obsmodel_colors.initialization(target, frame)

                    p_colors = obsmodel_colors.initialization(particles_c, frame)

                    # Calculate weights based on color vectors
                    weights_c = obsmodel_colors.likelihood(t_colors, p_colors, self.args.mu_c, self.args.sigma_c)

                    # Estimate weighted mean of all particles as final estimation
                    if len(weights_c[np.where(weights_c < self.args.thresh_c)]) != len(weights_c):
                        weights_c[np.where(weights_c < self.args.thresh_c)] = 0
                    if np.sum(weights_c, axis=0) != 0:
                        weights_c = weights_c/np.sum(weights_c)
                    else:
                        weights_c = 1 / len(weights_c) * np.ones(weights_c.shape)

                    est_c = np.dot(particles_c, weights_c).reshape(7,1)
                    states_clr = np.append(states_clr, est_c.reshape(1, 7), axis=0)

                    # Compute errors if ground truth is available
                    if gt_update:
                        err_c_euclidean = np.append(err_c_euclidean, self.calculate_error(GT, est_c, "euclidean"))
                        err_c_area_value, err_c_overlap_area_value = self.calculate_error(GT, est_c, "area")

                        err_c_area = np.append(err_c_area, err_c_area_value)
                        err_c_overlap_area = np.append(err_c_overlap_area, err_c_overlap_area_value)

                        if self.args.OM != "CM" and self.args.videoloss:
                            if self.args.error == "euclidean":
                                self.ploterror({"euclidean": err_c_euclidean}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "area":
                                self.ploterror({"area": err_c_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "overlap_area":
                                self.ploterror({"overlap_area": err_c_overlap_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "both":
                                self.ploterror({"euclidean": err_c_euclidean, "overlap_area": err_c_overlap_area}, self.args.OM, current_frame, self.video_images)
                            else:
                                raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

                    # Show bounding boxes of particles
                    if self.args.show_part:
                        for p in range(particles_c.shape[1]):
                            cv2.rectangle(frame, (int(particles_c[0, p]), int(particles_c[1, p])), (int(particles_c[0, p] + particles_c[4, p]), int(particles_c[1, p] + particles_c[5, p])), (0, 128, 255), 1)

                    # Show weighted average estimate
                    if self.args.OM != "CM":
                        cv2.rectangle(frame, (int(est_c[0]), int(est_c[1])),(int(est_c[0] + est_c[4]), int(est_c[1] + est_c[5])), (0, 255, 0), 2)


                # Fusion of both observation models - integrated method
                if self.args.OM == "CM":
                    dclr = self.calculate_error(target, est_c, "euclidean")
                    dmmt = self.calculate_error(target, est_m, "euclidean")

                    Wclr, Wmmt = self.fuse_states(self.args.beta, dclr, dmmt)

                    if self.args.bestofboth:
                        if self.args.error == "euclidean":
                            if err_m_euclidean[-1] >= err_c_euclidean[-1]:
                                Wmmt = 0
                            else:
                                Wclr = 0
                        elif self.args.error == "area" or self.args.error == "overlap_area":
                            if err_m_overlap_area[-1] <= err_c_overlap_area[-1]:
                                Wmmt = 0
                            else:
                                Wclr = 0
                        else:
                            raise ValueError("Invalid error type %s! Please choose either euclidean, area or overlap_area." % self.args.error)

                        Wclr = 1/(Wclr+Wmmt) * Wclr
                        Wmmt = 1 / (Wclr + Wmmt) * Wmmt

                    est_f = Wclr*est_c + Wmmt * est_m
                    states_fusion = np.append(states_fusion, est_f.reshape(1, 7), axis=0)

                    # Show estimate of fusion method
                    cv2.rectangle(frame, (int(est_f[0]), int(est_f[1])),(int(est_f[0] + est_f[4]), int(est_f[1] + est_f[5])), (0, 255, 0), 2)

                    # Compute errors if ground truth is available
                    if gt_update:
                        err_euclidean = np.append(err_euclidean, self.calculate_error(GT, est_f, "euclidean"))
                        err_area_value, err_overlap_area_value = self.calculate_error(GT, est_f, "area")

                        err_area = np.append(err_area, err_area_value)
                        err_overlap_area = np.append(err_overlap_area, err_overlap_area_value)

                        if self.args.videoloss:
                            if self.args.error == "euclidean":
                                self.ploterror({"euclidean": err_euclidean}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "area":
                                self.ploterror({"area": err_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "overlap_area":
                                self.ploterror({"overlap_area": err_overlap_area}, self.args.OM, current_frame, self.video_images)
                            elif self.args.error == "both":
                                self.ploterror({"euclidean": err_euclidean, "overlap_area": err_overlap_area}, self.args.OM, current_frame, self.video_images)
                            else:
                                raise ValueError("Invalid error type %s! Please choose either: euclidean, area or both" % self.args.error)

                # Show GT in frame
                if self.args.show_GT:
                    if gt_update:
                        cv2.rectangle(frame, (GT_x, GT_y), (GT_x_w, GT_y_h), color_GT, 2)

                # Show detection in frame
                if self.args.show_DT:
                    cv2.rectangle(frame, (int(target[0]), int(target[1])), (int(target[0] + target[4]), int(target[1] + target[5])), (0, 0, 255), 2)

                # Showing the output Image
                if self.args.show_frames:
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                imgs.append(frame)
                current_frame += 1


        # ============== Evaluation ================

        # Create videos of results
        if self.args.video:
            if self.args.videoloss:
                self.writevideoloss(imgs, self.args.OM)
            else:
                self.writevideo(imgs)

        shutil.rmtree(self.video_images)

        # Write statistics to file
        evaluation = Eval(self.args, self.evaluation_path)

        if not os.path.exists(os.path.join(self.evaluation_path, "error_plots")):
            os.makedirs(os.path.join(self.evaluation_path, "error_plots"))

        if not os.path.exists(os.path.join(self.evaluation_path, "error_files")):
            os.makedirs(os.path.join(self.evaluation_path, "error_files"))

        if self.args.OM.upper() == "MMT" or self.args.OM.upper() == "CM":
            self.ploterror({"euclidean": err_m_euclidean, "overlap_area": err_m_overlap_area}, "MMT", "all", os.path.join(self.evaluation_path, "error_plots"))
            self.errortotext({"euclidean": err_m_euclidean, "area": err_m_area, "overlap_area": err_m_overlap_area}, "MMT", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            self.statetotext(states_mmt, "MMT", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            evaluation.evalerror({"area": err_m_area, "euclidean": err_m_euclidean, "overlap_area": err_m_overlap_area}, "MMT", filename="ParticleFilter.xlsx")
        if self.args.OM.upper() == "CLR" or self.args.OM.upper() == "CM":
            self.ploterror({"euclidean": err_c_euclidean, "overlap_area": err_c_overlap_area}, "CLR", "all", os.path.join(self.evaluation_path, "error_plots"))
            self.errortotext({"euclidean": err_c_euclidean, "area": err_c_area, "overlap_area": err_c_overlap_area}, "CLR", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            self.statetotext(states_clr, "CLR", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            evaluation.evalerror({"area": err_c_area, "euclidean": err_c_euclidean, "overlap_area": err_c_overlap_area}, "CLR", filename="ParticleFilter.xlsx")
        if self.args.OM.upper() == "CM":
            self.ploterror({"euclidean": err_euclidean, "overlap_area": err_overlap_area}, "CM", "all", os.path.join(self.evaluation_path, "error_plots"))
            self.errortotext({"euclidean": err_euclidean, "area": err_area, "overlap_area": err_overlap_area}, "CM", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            self.statetotext(states_fusion, "CM", self.args.exp_nr, os.path.join(self.evaluation_path, "error_files"))
            evaluation.evalerror({"area": err_area, "euclidean": err_euclidean, "overlap_area": err_overlap_area}, "CM", filename="ParticleFilter.xlsx")

        print("finished.")


    def fuse_states(self, beta, dclr, dmmt):
        """
        Fusion method to match estimated states of different observation models
        :param beta: Factorization
        :param dclr: Distance between color histograms
        :param dmmt: Distance between moment invariants
        :return: Weights Wclr, Wmmt for fusion
        """
        Wclr = np.exp(-beta*dclr)/(np.exp(-beta*dclr) + np.exp(-beta*dmmt))
        Wmmt = np.exp(-beta*dmmt)/(np.exp(-beta*dclr) + np.exp(-beta*dmmt))

        return Wclr, Wmmt


    def writevideo(self, imgs):
        """
        Creates a video of annotated frames
        :param imgs: List of frames
        """
        video = cv2.VideoWriter(os.path.join(self.video_path, self.video_name), 0, 24, (imgs[0].shape[1], imgs[0].shape[0]))

        print("write video...")
        for i, frame in enumerate(imgs):
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print("video created.")


    def writevideoloss(self, imgs, type):
        """
        Creates video with annotated frames next to error curve
        :param imgs: List of frames
        :param type: String with type of error. Either 'euclidean', 'area' or 'overlap_area'
        """
        print("write video...")
        for i in range(len(imgs)):
            frame = imgs[i]

            filename = "Error_PF_" + str(type) + "_" + str(i+1) + ".jpg"
            try:
                error_plot = cv2.imread(os.path.join(self.video_images, filename), -1)
                # Resize image if necessary
                #error_plot = imutils.resize(error_plot, width=frame.shape[1], height=frame.shape[0])
                error_plot = imutils.resize(error_plot, width=720, height=error_plot.shape[0])
                frame = imutils.resize(frame, width=720, height=error_plot.shape[0])
            except:
                continue

            concat_frame = np.concatenate((frame, error_plot), axis=1)

            if i == 0:
                video = cv2.VideoWriter(os.path.join(self.video_path, self.video_name.replace(".mp4", "_error.mp4")), 0, 24, (concat_frame.shape[1], concat_frame.shape[0]))

            video.write(concat_frame)


        cv2.destroyAllWindows()
        video.release()

        print("video created.")


    def ploterror(self, error, OM, i, path):
        """
        Creates plots for errors
        :param error: Dictionary with error type as keys and numpy area of error values as value
        :param OM: String specifying observation model
        :param i: Frame number
        :param path: Path where plot will be saved
        """
        if str(i) == "all":
            filename = "Error_PF_" + str(OM) + "_" + str(i) + "_" + str(self.args.exp_nr) + ".jpg"
        else:
            filename = "Error_PF_" + str(OM) + "_" + str(i) + ".jpg"

        keys = list(error.keys())

        # Plot settings
        #plt.rc('font', family='serif', serif='Times')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('axes', labelsize=14)

        fig, axes = plt.subplots(nrows=len(keys), ncols=1)
        axes_array = np.array(axes)

        # width as measured in inkscape
        width = 1.5 * 3.487
        height = width / 1.618

        ylim_max = 1

        for i, ax in enumerate(axes_array.reshape(-1)):
            key = keys[i]
            value = error[key]
            if i == 0:
                color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                color = np.array([0.0, 0.0, 1.0])
            elif i == 2:
                color = np.array([0.0, 0.0, 1.0])
            else:
                color = np.random.rand()

            if key == "euclidean":
                ylim_min = 0
                ylim_max = max(ylim_max, np.max(value))
                ax.set_ylabel("Euclidean error")
                legend_str = ("Euclidean error",)
            elif key == "area":
                ylim_min = -1
                ylim_max = 1
                ax.set_ylabel("Area error")
                legend_str = ("Area error",)
            elif key == "overlap_area":
                ylim_min = 0
                ylim_max = 1
                ax.set_ylabel("Overlap rate")
                ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
                legend_str = ("Overlap rate",)
            else:
                raise ValueError("Invalid error type %s! Please choose either: euclidean or area" % key)


            ax.plot(value, color=color, linewidth=1.0)
            # ax.plot(value, color=color, linewidth=int(2), markersize=5)

            ax.legend(legend_str)
            if i == len(axes_array.reshape(-1)) -1 :
                ax.set_xlabel("Frames")
                ax.xaxis.set_ticks(np.arange(0, len(os.listdir(self.input_path)) + 1, 100), minor=True)

            ax.grid(linestyle='--', color='silver', which='both')

            ax.set_ylim([ylim_min, ylim_max])
            ax.set_xlim([0, len(os.listdir(self.input_path))])

        fig.suptitle("Error for Particle Filter")

        fig.subplots_adjust(left=.15, bottom=.16, right=.95, top=.92)
        fig.savefig(os.path.join(os.path.join(path, filename)), dpi=200)
        plt.close(fig)


    def errortotext(self, error, OM, run, filepath):
        """
        Writes error to .txt file
        :param error: Dictionary with numpy arrays of errors for each frame
        :param OM: Type of observation model used
        :param type: String with type of error. Either euclidean, area or overlap_area
        :param run: Number of runned experiment
        """
        print("Write loss for %s to .txt-file..." % OM)

        for type, error_val in error.items():
            filename = "Error_PF_" + str(OM) + "_" + str(type) + "_" + str(run) + ".txt"
            file_spec = os.path.join(filepath, filename)
            np.savetxt(file_spec, error_val)


    def statetotext(self, est_state, OM, run, filepath):
        """
        Writes estimated state vector to .txt file
        :param est_state: Numpy array of estimated state vectors for each frame - size: [frames x 7]
        :param OM: Type of observation model used
        :param run: Number of runned experiment
        """
        print("Write state_vector for %s to .txt-file..." % OM)

        filename = "States_PF_" + str(OM) + "_" + str(run) + ".txt"
        file_spec = os.path.join(filepath, filename)
        np.savetxt(file_spec, est_state)


    def calculate_error(self, GT, estimation, type):
        """
        Calculates the error between target and estimation. The variable type defines whether this calculation is based on fit of the
        areas of target and estimation or the euclidean distance between center point of target and estimation
        :param GT: Ground truth state vector
        :param estimation: Estimated state vector
        :param type: Specifies type of error. Either 'area' or 'euclidean'
        :return: error
        """
        if type == "area":
            # estimation in GT
            if (GT[0,:] <= estimation[0,:]) and (GT[1,:] <= estimation[1,:]) and (GT[0,:] + GT[4,:] >= estimation[0,:] + estimation[4,:]) and (GT[1,:] + GT[5,:] >= estimation[1,:] + estimation[5,:]):
                a = estimation[4,:]
                b = estimation[5,:]
                #upper_left = estimation[0:2,:]

            # GT in estimation
            elif (estimation[0,:] <= GT[0,:]) and (estimation[1,:] <= GT[1,:]) and (estimation[0,:] + estimation[4,:] >= GT[0,:] + GT[4,:]) and (estimation[1,:] + estimation[5,:] >= GT[1,:] + GT[5,:]):
                a = GT[4,:]
                b = GT[5,:]
                #upper_left = GT[0:2, :]

            else:
                a = -max(GT[0,:], estimation[0,:]) + min(GT[0,:] + GT[4,:], estimation[0,:] + estimation[4,:])
                b = -max(GT[1,:], estimation[1,:]) + min(GT[1,:] + GT[5,:], estimation[1,:] + estimation[5,:])
                if (a >= 0) and (b >= 0):
                    a = a
                    b = b
                else:
                    a = 0
                    b = 0
                #upper_left = np.array([max(GT[0,:], estimation[0,:]), max(GT[1,:], estimation[1,:])])

            # Calculate merging area
            A = a*b
            A_GT = GT[4,:]*GT[5,:]
            A_EST = estimation[4,:]*estimation[5,:]
            A_F = abs(A-A_EST)

            error_area = A/A_GT - A_F/A_EST

            A_union = (A_GT + A_F)
            error_overlap_area = A/A_union

            return error_area, error_overlap_area

        elif type == "euclidean":
            return np.linalg.norm(self.PF.centerparticles(estimation)[0:2, :] - self.PF.centerparticles(GT)[0:2, :])
        else:
            raise ValueError("Invalid type %s! Please choose either: area or euclidean." % type)


if __name__ == "__main__":
    # Paths
    input_frame_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dataset', 'frames'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'output'))

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Object Tracking for Pedestrians")

    # General configs for estimation
    parser.add_argument("--N", default=100, type=int, help="Specify number of particles")
    parser.add_argument("--alpha", default=0.0, type=float, help="Specify value fraction on how much we trust weighted estimate over initial target")
    parser.add_argument("--balance", default=True, type=bool_flag, help="Specify whether or not balance filtering should be used")
    parser.add_argument("--thresh_c", default=0.01, type=float, help="Specify estimation threshold for weights of color-based state vector")
    parser.add_argument("--thresh_m", default=0.01, type=float, help="Specify estimation threshold for weights of momentum-based state vector")
    parser.add_argument("--log_info", default=False, type=bool_flag, help="Specify whether to print information or not")
    parser.add_argument("--show_part", default=False, type=bool_flag, help="Specify whether to show bounding boxes of all particles")
    parser.add_argument("--show_frames", default=False, type=bool_flag, help="Specify whether to show every frame")
    parser.add_argument("--show_GT", default=True, type=bool_flag, help="Specify whether to show GT for every frame")
    parser.add_argument("--show_DT", default=False, type=bool_flag, help="Specify whether to show result of pedestrian detection for every frame")
    parser.add_argument("--error", default="overlap_area", type=str, help="Specify way of calculating the error between target and estimation. Choose either euclidean, area or both")
    parser.add_argument("--bestofboth", default=False, type=bool_flag, help="Specify whether to pick best estimation results for both methods, using CM")
    parser.add_argument("--exp_nr", default=1, type=int, help="Specify number of experiment")

    # Configs for detection
    parser.add_argument("--detect", default="HOG", type=str, help="Specify which detection method should be use. Choose HOG.")
    parser.add_argument("--colorbased", default=True, type=bool_flag, help="Specify whether detection should include color information")
    parser.add_argument("--color", default="BLACK", type=str, help="Specify color for detection based on color information")

    # Configs for motion model
    parser.add_argument("--dt", default=0.001, type=float, help="Specify time step")

    # Configs for observation model
    parser.add_argument("--OM", default="CLR", type=str, help="Specify which observation model to use. Choose either CLR, MMT or CM")
    parser.add_argument("--mu_c", default=0.0, type=float, help="Specify mean of color-based observation model")
    parser.add_argument("--sigma_c", default=0.5, type=float, help="Specify variance of color-based observation model")
    parser.add_argument("--mu_m", default=0.0, type=float, help="Specify mean of momentum-based observation model")
    parser.add_argument("--sigma_m", default=0.5, type=float, help="Specify variance of momentum-based observation model")
    parser.add_argument("--beta", default=0.5, type=float, help="Specify value for weight importance of fusion process")

    # Configs for resampling
    parser.add_argument("--resampling", default="SYS", type=str, help="Specify which resampling method to use. Choose either VAN or SYS")

    # Configs for video generation
    parser.add_argument("--video", default=True, type=bool_flag, help="Specify whether to create video from frames or not")
    parser.add_argument("--vid_name", default="", type=str, help="Specify name of video")
    parser.add_argument("--videoloss", default=False, type=bool_flag, help="Specify whether to show error in video")


    # Get arguments
    args = parser.parse_args()

    # Check Inputs
    if args.OM.upper() not in ["CLR", "MMT", "CM"]:
        raise ValueError("Please choose OM (Oberservation Model):\n- CLR\n- MMT\n- CM")

    # Run Tracker
    tracker = PedestrianTracker(args, input_frame_path, output_path)
    tracker.trackpedestrian()