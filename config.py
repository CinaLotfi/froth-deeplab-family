from pathlib import Path


class Config:
    """
    Global config for froth-deeplab-family.
    """

    # --------- paths ----------
    project_root = Path(__file__).resolve().parent

    data_root = project_root / "data"
    train_dir = data_root / "train"
    val_dir   = data_root / "test"   # using 'test' as val split

    outputs_root = project_root / "outputs"
    deeplab_out  = outputs_root / "deeplabv3_resnet101_finetune_out"

    # --------- dataset / labels ----------
    class_names = ["background", "froth"]
    num_classes = len(class_names)
    ignore_index = 255

    # optional: rasterized mask cache (None = no cache)
    mask_cache_dir = None   # e.g. project_root / "_mask_cache_v3_resnet101"
    mask_cache_overwrite = False

    # --------- image / preprocessing ----------
    resize_short_train = 1100
    resize_short_val   = 1024
    train_crop_size = 1024
    val_crop_size   = 1024

    # --------- dataloader ----------
    batch_size_train = 2
    batch_size_val   = 2
    num_workers = 0
    pin_memory = True

    # --------- optimization ----------
    epochs = 100
    accum_steps = 1
    use_amp = True
    head_lr = 1e-3
    backbone_lr = 1e-4
    weight_decay = 1e-4
    grad_clip_norm = 1.0
    power_poly = 0.9

    # --------- runtime ----------
    seed = 42
    device = "auto"      # "cuda" | "cpu" | "auto"
    resume_ckpt = None   # path to resume, or None

    @classmethod
    def setup(cls):
        """Create needed folders."""
        cls.data_root.mkdir(parents=True, exist_ok=True)
        cls.outputs_root.mkdir(parents=True, exist_ok=True)
        cls.deeplab_out.mkdir(parents=True, exist_ok=True)
