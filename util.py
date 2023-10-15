import numpy as np
import cv2
import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import copy
import pickle as pkl
from FFB6D.ffb6d.utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
import scipy.io as scio
from FFB6D.ffb6d.utils.basic_utils import Basic_Utils
from FFB6D.ffb6d.common import Config, ConfigRandLA
import open3d as o3d
from real_lab.perception.vision.realsense import RealSense
import matplotlib.pyplot as plt
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Init Realsense and Robot
"""
def init_camera():
    camera = RealSense(model="Realsense")
    return camera


def go_home(home, rtde_c):
    # rtde_c = rtde_control.RTDEControlInterface(robot_ip) #
    # rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip) #
    # home=[[-1.535,-1.7259,-0.777,-2.29,1.575,6.337,0.6,0.1,0]]
    rtde_c.moveJ(home)

"""
Grounding + SAM
"""

def init_GSAM_model(grounding_dino_config_path, grounding_dino_checkpoint_path, sam_encoder_version, sam_checkpoint_path):
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=grounding_dino_config_path, model_checkpoint_path=grounding_dino_checkpoint_path)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor

def get_annotate_image(classes, detections, image):
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    return annotated_frame

def nms_post_process(detections, nms_threshold):
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        nms_threshold
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    return detections

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_mask_image(image, detections):
    mask_image = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8) #
    for idx in range(detections.mask.shape[0]): #
        submask = copy.deepcopy(detections.mask)[idx] # (H,W)
        mask_image = np.bitwise_or(mask_image, submask.astype(np.uint8)) #
    mask_image = mask_image * 255 #
    return mask_image

# annotate image with detections
def get_annotate_detections_image(classes, detections, image):
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

"""
FFB6D
"""

def load_FFB6D_checkpoint(model=None, optimizer=None, filename="checkpoint"):

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec, model

def generate_pth_info(img_path, root_path, data_list_path):
    # obj_id is a list and the serial number in obj_id is correspond to the object number in the RGB image
    files = [os.path.join(img_path, file) for file in os.listdir(img_path) if 'color' in file]
    for idx in range(len(files)):
        files[idx] = files[idx].replace(root_path,'')
        files[idx] = files[idx].replace("-color.png",'')

    with open(data_list_path, '+w') as f:
        for id in range(len(files)):
            f.write(files[id])
            f.write("\n")

def generate_meta(obj_id, cam_intri, save_pth, factor_depth=10000, file_num="000001"):
    data = dict()
    data["obj_id"] = obj_id
    data["camera_intrinsic"] = cam_intri
    data["factor_depth"] = factor_depth
    data["poses"] = np.zeros((3,4,5)) # No GT, 0 padding
    file_path = os.path.join(save_pth, file_num+"-meta.mat")
    scio.savemat(file_path, data)

def cal_view_pred_pose(model, data, n_objects, full_mask, epoch=0, obj_id=-1): # estimate the obj pose
    model.eval()
    bs_utils = Basic_Utils(Config(ds_name='ycb'))
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys(): # convert to GPU tensor
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()

        ############### new adding code ###########################
        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        np_rgb = np_rgb[:, :, ::-1].copy()
        ############### new adding code ###########################

        # YCB evaluation
        pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
            cu_dt["cls_ids"].cpu().numpy()[0][0], cu_dt["choose"].cpu().numpy()[0], np_rgb.copy(),
            full_mask,
            pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
            end_points['pred_kp_ofs'][0], True, n_objects, True,
            None, None
        ) # add three new params: id, choose, ori_rgb

        # np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        # np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]

            obj_id = int(cls_id[0])
            mesh_pts_dataset = bs_utils.get_pointxyz(obj_id, ds_type='ycb').copy() # get origin pose in dataset

##############################################
            # Z=np.array([[-1,0,0],[0,1,0],[0,0,-1]])
            # pose = np.dot(Z, pose)
##############################################
#------------------------------------------------------------------------------------
            mesh_pts = np.dot(mesh_pts_dataset, pose[:, :3].T) + pose[:, 3] # predicted point cloud
            
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(mesh_pts)

            ################
            # source_pcd.rotate(Z)
            ##################
    
    return source_pcd, pose, mesh_pts_dataset, ori_rgb

def generate_target_pointcloud(depth_img, full_mask_img, cam_intri):
    process_img = depth_img * (full_mask_img/255)

    """
    first way to generate target point cloud
    """

    row, col = depth_img.shape #(480, 640)
    point_cloud = []

    dpt = process_img.astype(np.float32) / 10000  # conver to meters
    msk = (dpt > 1e-8).astype(np.float32)

    for v in range(row):
        for u in range(col):
            # if depth_img[v, u] == 0:
            #     continue
            depth = dpt[v, u]
            p_z = depth
            p_x = (u - cam_intri[0][2]) * p_z / cam_intri[0][0]
            p_y = (v - cam_intri[1][2]) * p_z / cam_intri[1][1]
            point_cloud.append([p_x, p_y, p_z])
    point_cloud = np.array(point_cloud)


    # dpt_3d = point_cloud.reshape(row, col, 3) * msk[:, :, None]
    # dpt_3d=dpt_3d.reshape(-1,3)#点云坐标

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # # Visualization point clouds
    # o3d.visualization.draw_geometries([pcd])

    """
    second way to generate target point cloud
    """

    # xmap = np.array([[j for i in range(640)] for j in range(480)])  # 480*640   高*宽
    # ymap = np.array([[i for i in range(640)] for j in range(480)])

    # dpt = process_img.astype(np.float32) / 10000  # 转为m
    # msk = (dpt > 1e-8).astype(np.float32)  # 1 和 0
    # row = (ymap - cam_intri[0][2]) * dpt / cam_intri[0][0]
    # col = (xmap - cam_intri[1][2]) * dpt / cam_intri[1][1]
    # dpt_3d = np.concatenate(
    #     (row[..., None], col[..., None], dpt[..., None]), axis=2
    # )
    # dpt_3d = dpt_3d * msk[:, :, None]
    # dpt_3d=dpt_3d.reshape(-1,3)#点云坐标

    # pcd = o3d.geometry.PointCloud()  # pcd类型的数据。

    # # 将点云转换成open3d中的数据形式并用pcd来保存，以方便用open3d处理
    # pcd.points = o3d.utility.Vector3dVector(dpt_3d)

    return pcd

def ICP(source, target, ICP_vis):
    # o3d.visualization.draw_geometries([source])
    # o3d.visualization.draw_geometries([target])
    #为两个点云上上不同的颜色
    source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
    target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

    # o3d.visualization.draw_geometries([source])
    # o3d.visualization.draw_geometries([target])

    # #为两个点云分别进行outlier removal，即离群点去除
    # processed_source, outlier_index = source.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)
    # processed_target, outlier_index = target.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)
    #下采样
    # processed_source = source.voxel_down_sample(voxel_size=0.02)
    # processed_target = target.voxel_down_sample(voxel_size=0.02)

    #o3d.visualization.draw_geometries([source,target])
    # o3d.visualization.draw_geometries([target])

    threshold = 0.05  #移动范围的阀值


    trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                            [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                            [0,0,1,0],   # 这个矩阵为初始变换
                            [0,0,0,1]])
    #运行icp
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    #将我们的矩阵依照输出的变换矩阵进行变换
    print(reg_p2p)    #输出配准的结果准确度等信息
    print("Transformation is:")
    print(reg_p2p.transformation)  # 打印旋转矩阵
    source.transform(reg_p2p.transformation)

    if ICP_vis:
        #创建一个 o3d.visualizer class
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source)
        vis.add_geometry(target)
        threshold = 1.0
        icp_iteration = 300
        save_image = False

        for i in range(icp_iteration):
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source, target, threshold, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
            source.transform(reg_p2l.transformation)
            vis.update_geometry(source)
            vis.poll_events()
            vis.update_renderer()
            if save_image:
                vis.capture_screen_image("temp_%04d.jpg" % i)
        vis.destroy_window()

    return reg_p2p.transformation

def visual_pose_estimation(cam_intri, final_pose, mesh_pts_dataset, obj_id, np_rgb):
    bs_utils = Basic_Utils(Config(ds_name='ycb'))

    mesh_pts = np.dot(mesh_pts_dataset, final_pose[:3, :3].T) + final_pose[:3, 3]

    mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, cam_intri)#二维点
    color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
    np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)

    cv2.imwrite("/home/barry/cxg/YCB_pose_estimation/fine_pose_estimation.png", np_rgb)  #存下图片

    return mesh_pts
