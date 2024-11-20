OUTPUT_DIR = 'output/NOCS_REAL/OpenSourceCheck_1710996986'
EXP_NAME = ''
DEBUG = False
SEED = 1710996986
CUDNN_BENCHMARK = True
IM_BACKEND = 'cv2'
VIS_PERIOD = 0
INPUT = dict(
    FORMAT='BGR',
    MIN_SIZE_TRAIN=(480, ),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING='choice',
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=True,
    BP_DEPTH=False,
    AUG_DEPTH=True,
    NORM_DEPTH=False,
    DROP_DEPTH_RATIO=0.2,
    DROP_DEPTH_PROB=0.5,
    ADD_NOISE_DEPTH_LEVEL=0.01,
    ADD_NOISE_DEPTH_PROB=0.9,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE='ROI10D',
    COLOR_AUG_CODE='',
    COLOR_AUG_SYN_ONLY=False,
    RANDOM_FLIP='none',
    WITH_BG_DEPTH=False,
    BG_DEPTH_FACTOR=10000.0,
    BG_TYPE='VOC_table',
    BG_IMGS_ROOT='datasets/VOCdevkit/VOC2012/',
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.0,
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    DZI_TYPE='uniform',
    DZI_PAD_SCALE=1.0,
    DZI_SCALE_RATIO=0.25,
    DZI_SHIFT_RATIO=0.25,
    SMOOTH_XYZ=False,
    ALIGN_PCL=False,
    WITH_IMG=False,
    PCL_WITH_COLOR=False,
    SAMPLE_DEPTH_FROM_BALL=True,
    DEPTH_SAMPLE_BALL_RATIO=0.6,
    FPS_SAMPLE=False,
    MAX_SYM_DISC_STEP=0.01,
    BBOX_TYPE_TEST='est',
    INIT_POSE_TYPE_TRAIN=['gt_noise'],
    INIT_SCALE_TYPE_TRAIN=['gt_noise'],
    INIT_POSE_TRAIN_PATH=
    'datasets/NOCS/train_init_poses/init_with_last_frame.pkl',
    INIT_POSE_TYPE_TEST='est',
    NOISE_ROT_STD_TRAIN=(10, 5, 2.5, 1.25),
    NOISE_ROT_STD_TEST=15,
    NOISE_ROT_MAX_TRAIN=45,
    NOISE_ROT_MAX_TEST=45,
    NOISE_TRANS_STD_TRAIN=[(0.02, 0.02, 0.02), (0.01, 0.01, 0.01),
                           (0.005, 0.005, 0.005)],
    NOISE_TRANS_STD_TEST=[(0.01, 0.01, 0.005), (0.01, 0.01, 0.01),
                          (0.005, 0.005, 0.01)],
    INIT_TRANS_MIN_Z=0.1,
    NOISE_SCALE_STD_TRAIN=[(0.01, 0.01, 0.01), (0.005, 0.005, 0.005),
                           (0.002, 0.002, 0.002)],
    NOISE_SCALE_STD_TEST=[(0.001, 0.005, 0.001), (0.005, 0.001, 0.005),
                          (0.01, 0.01, 0.01)],
    INIT_SCALE_MIN=0.04,
    RANDOM_TRANS_MIN=[-0.35, -0.35, 0.5],
    RANDOM_TRANS_MAX=[0.35, 0.35, 1.3],
    RANDOM_SCALE_MIN=[0.04, 0.04, 0.04],
    RANDOM_SCALE_MAX=[0.5, 0.3, 0.4],
    MEAN_MODEL_PATH=
    'datasets/NOCS/obj_models/cr_normed_mean_model_points_spd.pkl',
    KPS_TYPE='mean_shape',
    USE_CMRA_MODEL=True,
    WITH_NEG_AXIS=False,
    BBOX3D_AUG_PROB=0.5,
    RT_AUG_PROB=0.5,
    DEPTH_BILATERAL_FILTER_TEST=False,
    DEPTH_HOLE_FILL_TEST=False,
    NUM_KPS=512,
    NUM_PCL=512,
    ZERO_CENTER_INPUT=True,
    NOCS_DIST_THER=0.03,
    CANONICAL_ROT=[(1, 0, 0, 0.5), (0, 0, 1, -0.7)],
    CANONICAL_TRANS=[0, 0, 1.0],
    CANONICAL_SIZE=[0.2, 0.2, 0.2],
    OCCLUDE_MASK_TEST=False,
    WITH_PCL=True)
DATASETS = dict(
    TRAIN=('nocs_train_real', ),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    DATA_LEN_WITH_TRAIN2=True,
    PROPOSAL_FILES_TRAIN=(),
    PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
    TEST=('nocs_test_real', ),
    PROPOSAL_FILES_TEST=(),
    PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    DET_FILES_TRAIN=(),
    DET_TOPK_PER_OBJ_TRAIN=1,
    DET_THR_TRAIN=0.0,
    DET_FILES_TEST=(),
    DET_TOPK_PER_OBJ=1,
    DET_THR=0.0,
    INIT_POSE_FILES_TEST=(
        'datasets/NOCS/test_init_poses/init_pose_spd_nocs_real.json', ),
    INIT_POSE_TOPK_PER_OBJ=1,
    INIT_POSE_THR=0.0)
DATALOADER = dict(
    NUM_WORKERS=32,
    PERSISTENT_WORKERS=False,
    MAX_OBJS_TRAIN=1000,
    ASPECT_RATIO_GROUPING=False,
    SAMPLER_TRAIN='TrainingSampler',
    REPEAT_THRESHOLD=0.0,
    FILTER_EMPTY_ANNOTATIONS=True,
    FILTER_EMPTY_DETS=True,
    FILTER_VISIB_THR=0.0,
    REMOVE_ANNO_KEYS=[])
SOLVER = dict(
    IMS_PER_BATCH=12,
    REFERENCE_BS=12,
    TOTAL_EPOCHS=1,
    OPTIMIZER_CFG=dict(type='Ranger', lr=0.0001, weight_decay=0),
    GAMMA=0.1,
    BIAS_LR_FACTOR=1.0,
    LR_SCHEDULER_NAME='flat_and_anneal',
    WARMUP_METHOD='linear',
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    ANNEAL_METHOD='cosine',
    ANNEAL_POINT=0.72,
    POLY_POWER=0.9,
    REL_STEPS=(0.5, 0.75),
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
    CLIP_GRADIENTS=dict(
        ENABLED=False, CLIP_TYPE='value', CLIP_VALUE=1.0, NORM_TYPE=2.0),
    SET_NAN_GRAD_TO_ZERO=False,
    AMP=dict(ENABLED=False),
    WEIGHT_DECAY=0,
    OPTIMIZER_NAME='Ranger',
    BASE_LR=0.0001,
    MOMENTUM=0.9)
TRAIN = dict(PRINT_FREQ=100, VERBOSE=False, VIS=False, VIS_IMG=False)
VAL = dict(
    DATASET_NAME='nocs',
    RESULTS_PATH='',
    ERROR_TYPES='ad,rete,re,te,proj',
    N_TOP=1,
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=False,
    SAVE_BOP_CSV_ONLY=False)
TEST = dict(
    EVAL_PERIOD=1,
    VIS=False,
    TEST_BBOX_TYPE='gt',
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    AMP_TEST=False,
    SAVE_RESULTS_ONLY=False)
DIST_PARAMS = dict(backend='nccl')
MODEL = dict(
    DEVICE='cuda',
    WEIGHTS='./output/NOCS_REAL/OpenSourceCheck_1710996986/new_model.pth',
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_POSES_TEST=True,
    REFINE_SCLAE=True,
    GeoReF=dict(
        NAME='GeoReF',
        TASK='refine',
        REFINE_SCLAE=True,
        NUM_CLASSES=6,
        N_ITER_TRAIN=4,
        N_ITER_TRAIN_WARM_EPOCH=4,
        N_ITER_TEST=4,
        USE_MTL=False,
        BACKBONE=dict(FREEZE=True),
        GCN3D=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type='gcn3d', num_points=1532, neighbor_num=10,
                support_num=7)),
        FEATNET=dict(
            FREEZE=False,
            INIT_CFG=dict(num_points=1024, global_feat=False, out_dim=1024)),
        KPSNET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type='point_net', num_points=8, global_feat=True,
                out_dim=1024),
            LR_MULT=1.0),
        POSE_HEAD=dict(
            FREEZE=False,
            ROT_TYPE='ego_rot6d',
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type='FC_RotTransHead',
                in_dim=2048,
                num_layers=2,
                feat_dim=256,
                norm='none',
                num_gn_groups=32,
                act='gelu'),
            LR_MULT=1.0,
            DELTA_T_SPACE='image',
            DELTA_T_WEIGHT=1.0,
            T_TRANSFORM_K_AWARE=True,
            DELTA_Z_STYLE='cosypose'),
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_TYPE='ego_rot6d',
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type='ConvOutPerRotHead',
                in_dim=1088,
                rot_dim=3,
                feat_flatten=False,
                norm_input=False,
                num_points=1024,
                point_bias=True,
                num_layers=2,
                kernel_size=1,
                feat_dim=256,
                norm='GN',
                num_gn_groups=32,
                act='gelu'),
            LR_MULT=1.0,
            DELTA_T_SPACE='image',
            DELTA_T_WEIGHT=1.0,
            T_TRANSFORM_K_AWARE=True,
            DELTA_Z_STYLE='cosypose',
            SCLAE_TYPE='iter_add'),
        NOCS_HEAD=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type='ConvPointNocsHead',
                in_dim=1286,
                feat_kernel_size=1,
                out_kernel_size=1,
                num_layers=2,
                feat_dim=256,
                norm='GN',
                num_gn_groups=32,
                act='gelu',
                last_act='sigmoid',
                norm_input=False,
                use_bias=False)),
        T_HEAD=dict(
            WITH_KPS_FEATURE=True,
            FREEZE=False,
            INIT_CFG=dict(type='FC_TransHead', in_dim=1286),
            LR_MULT=1.0),
        S_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(type='FC_SizeHead', in_dim=1286),
            LR_MULT=1.0),
        TS_HEAD=dict(
            WITH_KPS_FEATURE=False,
            WITH_INIT_SCALE=True,
            WITH_INIT_TRANS=False,
            FREEZE=False,
            INIT_CFG=dict(
                type='FC_TransSizeHead',
                in_dim=1091,
                num_layers=2,
                kernel_size=1,
                feat_dim=256,
                norm='GN',
                num_gn_groups=32,
                act='gelu',
                num_points=1024,
                norm_input=False),
            LR_MULT=1.0),
        LOSS_CFG=dict(
            PM_LOSS_TYPE='L1',
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=True,
            PM_R_ONLY=True,
            PM_WITH_SCALE=True,
            PM_DISENTANGLE_T=False,
            PM_DISENTANGLE_Z=False,
            PM_T_USE_POINTS=True,
            PM_USE_BBOX=False,
            PM_LW=1.0,
            ROT_LOSS_TYPE='angular',
            ROT_YAXIS_LOSS_TYPE='L1',
            ROT_LW=1.0,
            ROT_LW_PP=0.0,
            TRANS_LOSS_TYPE='L1',
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=1.0,
            SCALE_LOSS_TYPE='L1',
            SCALE_LW=1.0,
            NOCS_LOSS_TYPE='L1',
            NOCS_LW=0.0,
            SYM_NOCS_TYPE='YAXIS',
            NOCS_SYM_AWARE=True,
            PM_NORM_BY_EXTENT=False)),
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False)
EXP_ID = 'real275_test'
RESUME = False