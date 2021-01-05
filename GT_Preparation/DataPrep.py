import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import cv2

# ====================================================================================
#  Prepare Ground truth data - Create annotated video of frames with GT information
# ====================================================================================

class PrepareGT:
    """
    Prepares the GroundTruth data by creating videos, annotating BBoxes, etc.
    """
    def __init__(self, input_path, output_path, video_name = "gt.mp4"):
        self.input_path = input_path
        self.output_path = output_path
        self.video_name = video_name


    def createvid(self, annotating=False, gt_filename = "gt.txt"):
        """
        Creates a video given a sequence of frames.
        :param annotating: If True, BBoxes are shown in video
        :param gt_filename: Filename of ground truth data
        """
        if not os.path.exists(self.input_path):
            raise OSError("Input path %s is invalid!", self.input_path)

        if annotating:
            color = (255,0,0)
            bboxes = open(os.path.join(self.input_path, gt_filename), "r")
            positions = bboxes.readlines()
            positions = [p.replace("\n", "").split(",") for p in positions]

        imgs = []

        print("iterate over frames...")

        # Iterate over all frames
        i = 0
        for file in os.listdir(self.input_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                frame_spec = os.path.join(self.input_path, file)

                if i % 100 == 0:
                    print("Frames processed: %d/%d" % (i, len(os.listdir(self.input_path))))

                # Read frame
                frame = cv2.imread(frame_spec, -1)

                if i == 0:
                    height, width, layers = frame.shape

                if annotating:
                    if i <= len(positions):
                        # draw rectangle
                        try:
                            cv2.rectangle(frame, (round(float(positions[i][0])), round(float(positions[i][1]))), (round(float(positions[i][2])), round(float(positions[i][3]))), color, 2)
                        except:
                            pass

                # Append frame
                imgs.append(frame)
                i += 1

        # Write video
        video = cv2.VideoWriter(os.path.join(self.output_path, self.video_name), 0, 24, (width, height))

        print("write video...")
        for frame in imgs:
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
        print("finished.")



if __name__ == "__main__":
    # Define input and output path
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dataset', 'frames'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dataset', 'videos', 'generated'))
    annotate = True

    if annotate:
        video_name = "gt_annotated.mp4"
    else:
        video_name = "gt.mp4"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_prep = PrepareGT(input_path, output_path, video_name)
    data_prep.createvid(annotate)
