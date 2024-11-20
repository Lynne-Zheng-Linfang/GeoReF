_base_ = ["../../_base_/georef_base.py"]

# SEED =1710996986 
# OUTPUT_DIR = "output/NOCS_REAL/OpenSourceCheck_" #new generated dataset
OUTPUT_DIR = "output/NOCS_REAL/OpenSourceCheckMugHandleMeta_" #new generated dataset
INPUT = dict(
    COLOR_AUG_PROB=0.0,
    DEPTH_SAMPLE_BALL_RATIO=0.6,
    BBOX_TYPE_TEST="est",  # from_pose | est | gt | gt_aug (TODO)
    INIT_POSE_TYPE_TRAIN=["gt_noise"],  # gt_noise | random | canonical
    NOISE_ROT_STD_TRAIN=(10, 5, 2.5, 1.25),  # randomly choose one
    NOISE_TRANS_STD_TRAIN=[
        (0.02, 0.02, 0.02),
        (0.01, 0.01, 0.01),
        (0.005, 0.005, 0.005),
    ],
    NOISE_SCALE_STD_TRAIN=[
        (0.01, 0.01, 0.01),
        (0.005, 0.005, 0.005),
        (0.002, 0.002, 0.002),
    ],
    INIT_POSE_TYPE_TEST="est",  # gt_noise | est | canonical
    KPS_TYPE="mean_shape",  # bbox_from_scale | mean_shape |fps (abla)
    WITH_DEPTH=True,
    AUG_DEPTH=True,
    WITH_PCL=True,
    WITH_IMG=False,
    BP_DEPTH=False,
    NUM_KPS=512,
    NUM_PCL=512,
    # augmentation when training
    BBOX3D_AUG_PROB=0.5,
    RT_AUG_PROB=0.5,
    # pose focalization
    ZERO_CENTER_INPUT=True,
)

DATALOADER = dict(
    NUM_WORKERS=32,
)

SOLVER = dict(
    IMS_PER_BATCH=12,
    TOTAL_EPOCHS=150,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("nocs_train_real",),
    TEST=("nocs_test_real",),
    INIT_POSE_FILES_TEST=("datasets/NOCS/test_init_poses/init_pose_spd_nocs_real.json",),
)

MODEL = dict(
    LOAD_POSES_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    REFINE_SCLAE=True,
    GeoReF=dict(
        NAME="GeoReF",  # used module file name (define different model types)
        TASK="refine",  # refine | init | init+refine
        NUM_CLASSES=6,  # only valid for class aware
        N_ITER_TRAIN=4,
        N_ITER_TRAIN_WARM_EPOCH=4,  # linearly increase the refine iter from 1 to N_ITER_TRAIN until this epoch
        N_ITER_TEST=4,
        FEATNET=dict( # TODO: Note that here might can be changed.
            FREEZE=False,
            INIT_CFG=dict(
                num_points=1024,
                global_feat=False,
                out_dim=1024, # Final out will includ the global feat, which will be 1024+64
            ),
        ),
        ## disentangled pose head for delta R/T/s
        ROT_HEAD=dict(
            ROT_TYPE="ego_rot6d",  # {ego|allo}_rot6d
            INIT_CFG=dict(
                type="ConvOutPerRotHead",
                in_dim=1088,
                num_layers=2,
                kernel_size=1,
                feat_dim=256,
                norm="GN",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
                num_points=1024,
                rot_dim=3,  # ego_rot6d
                norm_input=False,
            ),
            SCLAE_TYPE="iter_add",
        ),
        TS_HEAD=dict(
            WITH_KPS_FEATURE=False,
            WITH_INIT_SCALE=True,
            INIT_CFG=dict(
                type="FC_TransSizeHead",
                in_dim=1088 + 3,
                num_layers=2,
                kernel_size=1,
                feat_dim=256,
                norm="GN",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
                num_points=1024,
                norm_input=False,
            ),
        ),
        LOSS_CFG=dict(
            # point matching loss ----------------
            PM_LOSS_SYM=True,  # use symmetric PM loss
            PM_NORM_BY_EXTENT=False,  # 1. / extent.max(1, keepdim=True)[0]
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_WITH_SCALE=True,
            PM_LW=1.0,
            # rot loss --------------
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_LW=1.0,
            ROT_YAXIS_LOSS_TYPE="L1",
            # trans loss -----------
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=1.0,
            # scale loss ----------------------------------
            SCALE_LOSS_TYPE="L1",
            SCALE_LW=1.0,
        ),
    ),
)
