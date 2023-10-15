import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import cv2
import argparse
import os
import torch
import copy
from util import *
# from tsdf import *
# from real_test_hand import *
# from real_test2 import *

import sys
# sys.path.append("/home/barry/cxg/pose_estimation/FFB6D/ffb6d")
from FFB6D.ffb6d.common import Config, ConfigRandLA
from FFB6D.ffb6d.models.ffb6d import FFB6D
from FFB6D.ffb6d.datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
import tqdm
import time
from manopth.manolayer import ManoLayer
from torch.utils.data import DataLoader
import torch.nn.functional as F

FILE_NUM="000001"
config = Config(ds_name='ycb')

if __name__=="__main__":
    parser = argparse.ArgumentParser("Hand-Object Pose Estimation", add_help=True)
    # -------------------- Realsense and Robot configure ----------------
    parser.add_argument("--save_cam_intri_path", type= str,
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/real_img/cam_intrin.txt")
    parser.add_argument("--use_record_data", type=bool, default=True)
    # -------------------- GSAM model configure --------------------
    parser.add_argument("--grounding_dino_config_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_dino_checkpoint_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_encoder_version", type=str, default="vit_h")
    parser.add_argument("--sam_checkpoint_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.8)
    # -------------------- Segmentation input configure --------------------
    parser.add_argument("--source_image_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/real_img/000001-color.png")
    parser.add_argument("--source_depth_path", type=str,
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/real_img/000001-depth.png")
    parser.add_argument("--save_mask_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/real_img/000001-label.png")
    parser.add_argument("--classes", type=list, default=["scissors"], help="['object1', 'object2', ....]")
    # -------------------- FFB6D model configure --------------------
    parser.add_argument("--FFB6D_checkpoint", type=str, 
                        default='/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/train_log/ycb/checkpoints/FFB6D_best.pth.tar')
    parser.add_argument("--n_objects", type=int, default=21+1, help="21 objects + background")
    parser.add_argument("--n_sample_points", type=int, default=480 * 640 // 24, help="Number of input points")
    parser.add_argument("--n_keypoints", type=int, default=8)
    parser.add_argument("--img_path", type=str, 
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/real_img",
                        help="The root path to input data")
    parser.add_argument("--obj_id", type=int, default=17, 
                        help="Refer to /home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/dataset_config/classes.txt")
    parser.add_argument("--real_data_list", type=str,
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/dataset_config/real_data_list.txt",
                        help="Store the real data filename")
    parser.add_argument("--ffb6d_root", type=str,
                        default='/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/')
    parser.add_argument("--mesh_path", type=str,
                        default="/home/barry/cxg/YCB_pose_estimation/FFB6D/ffb6d/datasets/ycb/YCB_Video_Dataset/models")
    parser.add_argument("--ICP_vis", type=bool, default=False, help="visualize the process of ICP")
    parser.add_argument("--vis_pose", type=bool, default=False, help="visualize the result of object pose estimation in the view of 2D")
    args = parser.parse_args()

    


    """
    Init camera and robot
    """
    start1 = time.time()
    if not args.use_record_data:
        camera = init_camera()
        # go_home(args.home, rtde_c)
        cam_intri = camera.get_camera_params()['cam_intri']
        time.sleep(1)
        rgb_img, depth_img = camera.get_data()
        cv2.imwrite(args.source_image_path, rgb_img)
        cv2.imwrite(args.source_depth_path, np.round(depth_img * 10000).astype(np.uint16))
        np.savetxt(args.save_cam_intri_path, cam_intri)

    else:
        cam_intri = np.loadtxt(os.path.join(args.img_path, "cam_intrin.txt"))
        rgb_img = cv2.imread(args.source_image_path)
    time.sleep(0.5)
    generate_meta(args.obj_id, cam_intri, args.img_path)
    depth_img = cv2.imread(os.path.join(args.img_path, "000001-depth.png"), -1)

    """
    Get the segmentation information by using GSAM
    """
    # init model
    grounding_dino_model, sam_predictor = init_GSAM_model(args.grounding_dino_config_path, args.grounding_dino_checkpoint_path,
                                                     args.sam_encoder_version, args.sam_checkpoint_path)
    
    end1 = time.time()
    print("Load model time:%s" %(end1-start1))

    # for _ in range(2):
    start3 = time.time()
    # load image
    image = cv2.imread(args.source_image_path)

    start_det = time.time()
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=args.classes,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    ) # It will be more time-consuming to load and use it for the first time.
    print("detect time:%s" %(time.time()-start_det))

    # # get annotated image
    # annotated_frame = get_annotate_image(args.classes, detections, image)
    # # save the annotated grounding dino image
    # cv2.imwrite("groundingdino_annotated_rgb4.jpg", annotated_frame)

    start_NMS = time.time()
    # NMS post process
    detections = nms_post_process(detections, args.nms_threshold)
    print("NMS time:%s" %(time.time()-start_NMS))

    start_seg = time.time()
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    print("seg time:%s" %(time.time()-start_seg))

    # get mask image
    mask_image = get_mask_image(image, detections) # image * 255
    cv2.imwrite(args.save_mask_path, mask_image) #
    
    end3 = time.time()
    print("Segmentation time:%s" %(end3-start3))

    # # get annotate image with detections
    # annotated_image = get_annotate_detections_image(args.classes, detections, image)
    # cv2.imwrite("grounded_sam_annotated_rgb4.jpg", annotated_image)

    """
    Estimate the object pose 
    """
    start2 = time.time()
    rndla_cfg = ConfigRandLA
    model = FFB6D(n_classes=args.n_objects, n_pts=args.n_sample_points, rndla_cfg=rndla_cfg, n_kps=args.n_keypoints)
    model.cuda()

    # load status from checkpoint
    _, _, _, model = load_FFB6D_checkpoint(model, None, filename=args.FFB6D_checkpoint)

    # load input data
    generate_pth_info(args.img_path, args.ffb6d_root, args.real_data_list)
    input_data = YCB_Dataset('owner', cam_intri)
    obj_id_class = -1 # 
    input_loader = torch.utils.data.DataLoader(input_data, batch_size=1, shuffle=False)

    source_pd = []
    for i, data in tqdm.tqdm(enumerate(input_loader), leave=False, desc="val"):
    # leave: End is whether to keep the progress bar, the default is True # desc: progress bar title
        source_point, pose, mesh_pts_dataset, ori_rgb = cal_view_pred_pose(model, data, args.n_objects, mask_image, epoch=i, obj_id=obj_id_class) # predicted obj point cloud
        source_pd.append(source_point)

    target = generate_target_pointcloud(depth_img, mask_image, cam_intri)

    # ICP refine pose estimation
    ICP_matrix = ICP(source_pd[0], target, args.ICP_vis)

    b = np.array([[0, 0, 0,1]])
    pose = np.r_[pose, b] # predicted  

    # Z=np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    # pose = np.dot(Z,pose)
    final_pose=np.dot(ICP_matrix,pose) # final obj pose estimation        

    if args.vis_pose:
        mesh_pts = visual_pose_estimation(cam_intri, final_pose, mesh_pts_dataset, args.obj_id, ori_rgb) # final obj point cloud in camera coordinate system
    
    end2 = time.time()
    print("Object pose estimation time:%s"%(end2-start2))