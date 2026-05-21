"""
create_bevdet_pkl_trainval.py

Создаёт PKL-файл в формате BEVDet из pseudo-LiDAR данных и nuScenes метаданных.
Версия для полного v1.0-trainval с дефолтными путями под новую директорию
pseudo_lidar_v16_trainval (сгенерированную скриптом v16_trainval).

Отличие от create_bevdet_pkl.py: только дефолтные значения аргументов.
Логика работы идентична.

Запуск:
    conda activate bevdet
    python scripts/create_bevdet_pkl_trainval.py \
        --dataroot /home/max/Phd/Phase1/dataset/v1.0-trainval \
        --pseudo_lidar_dir /home/max/Phd/Phase1/dataset/pseudo_lidar_v16_trainval \
        --version v1.0-trainval \
        --out /home/max/Phd/Phase1/dataset/pseudo_lidar_v16_trainval/bevdet_infos.pkl
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

CAMERA_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'movable_object.trafficcone': 'traffic_cone',
    'static_object.bicycle_rack': 'ignore',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'animal': 'ignore',
}


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = nusc.get_sample_data_path(sd_rec['token'])

    l2e_r_s_mat = Quaternion(cs_record['rotation']).rotation_matrix
    e2g_r_s_mat = Quaternion(pose_record['rotation']).rotation_matrix
    l2e_t_s = np.array(cs_record['translation'])
    e2g_t_s = np.array(pose_record['translation'])

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    return {
        'data_path': data_path,
        'type': sd_rec['channel'],
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
        'sensor2lidar_rotation': R.T,
        'sensor2lidar_translation': T,
    }


def get_gt(info, nusc):
    sample = nusc.get('sample', info['token'])
    ann_infos = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        velocity = nusc.box_velocity(ann_token)
        if np.any(np.isnan(velocity)):
            velocity = np.zeros(3)
        ann['velocity'] = velocity
        ann_infos.append(ann)

    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT']['ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse

    gt_boxes, gt_labels = [], []
    for ann_info in ann_infos:
        cat = NAME_MAPPING.get(ann_info['category_name'], 'ignore')
        if cat not in CLASS_NAMES:
            continue
        if ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0:
            continue
        from nuscenes.utils.data_classes import Box
        box = Box(ann_info['translation'], ann_info['size'],
                  Quaternion(ann_info['rotation']),
                  velocity=ann_info['velocity'])
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_boxes.append(np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo]))
        gt_labels.append(CLASS_NAMES.index(cat))

    return gt_boxes, gt_labels


def build_infos(nusc, pseudo_lidar_dir, split_scenes):
    pseudo_lidar_dir = Path(pseudo_lidar_dir)
    infos = []
    skipped = 0

    for i, sample in enumerate(nusc.sample):
        scene = nusc.get('scene', sample['scene_token'])
        if scene['name'] not in split_scenes:
            continue

        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_filename = Path(lidar_sd['filename']).name
        pseudo_path = pseudo_lidar_dir / 'samples' / 'LIDAR_TOP' / lidar_filename

        if not pseudo_path.exists():
            skipped += 1
            continue

        cs_record = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

        l2e_t = np.array(cs_record['translation'])
        l2e_r_mat = Quaternion(cs_record['rotation']).rotation_matrix
        e2g_t = np.array(pose_record['translation'])
        e2g_r_mat = Quaternion(pose_record['rotation']).rotation_matrix

        info = {
            'lidar_path': str(pseudo_path),
            'token': sample['token'],
            'sweeps': [],
            'cams': {},
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
        }

        for cam in CAMERA_TYPES:
            cam_token = sample['data'][cam]
            cam_sd = nusc.get('sample_data', cam_token)
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat)
            cam_info['cam_intrinsic'] = cam_intrinsic
            info['cams'][cam] = cam_info

        annotations = [nusc.get('sample_annotation', t) for t in sample['anns']]
        from nuscenes.utils.data_classes import Box
        _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array([nusc.box_velocity(t)[:2] for t in sample['anns']])
        for j in range(len(boxes)):
            velo = np.array([*velocity[j], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[j] = velo[:2]

        names = []
        for b in boxes:
            name = NAME_MAPPING.get(b.name, b.name)
            names.append(name)
        names = np.array(names)

        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = np.array(
            [(a['num_lidar_pts'] + a['num_radar_pts']) > 0 for a in annotations], dtype=bool)

        info['ann_infos'] = get_gt(info, nusc)
        info['occ_path'] = None

        infos.append(info)

        if (i + 1) % 100 == 0:
            print(f"  обработано {i+1}/{len(nusc.sample)}, добавлено {len(infos)}, пропущено {skipped}")

    return infos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default='/home/max/Phd/Phase1/dataset/v1.0-trainval')
    p.add_argument('--pseudo_lidar_dir',
                   default='/home/max/Phd/Phase1/dataset/pseudo_lidar_v16_trainval')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--out',
                   default='/home/max/Phd/Phase1/dataset/pseudo_lidar_v16_trainval/bevdet_infos.pkl')
    args = p.parse_args()

    print(f"Загрузка NuScenes ({args.version}) из {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    splits = create_splits_scenes()
    train_scenes = set(splits['train'])
    val_scenes = set(splits['val'])

    print(f"Сцен train: {len(train_scenes)}, val: {len(val_scenes)}")
    print(f"Всего сэмплов в датасете: {len(nusc.sample)}")

    print("\nСборка train infos...")
    train_infos = build_infos(nusc, args.pseudo_lidar_dir, train_scenes)
    print(f"Train: {len(train_infos)} сэмплов")

    print("\nСборка val infos...")
    val_infos = build_infos(nusc, args.pseudo_lidar_dir, val_scenes)
    print(f"Val: {len(val_infos)} сэмплов")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_out = out_path.parent / (out_path.stem + '_train.pkl')
    val_out = out_path.parent / (out_path.stem + '_val.pkl')

    with open(train_out, 'wb') as f:
        pickle.dump({'infos': train_infos, 'metadata': {'version': args.version}}, f)
    print(f"\nTrain PKL: {train_out}")

    with open(val_out, 'wb') as f:
        pickle.dump({'infos': val_infos, 'metadata': {'version': args.version}}, f)
    print(f"Val PKL:   {val_out}")


if __name__ == '__main__':
    main()
