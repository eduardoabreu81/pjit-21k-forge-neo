# P-JiT 21K support for Forge Neo
# A setting for the noise upstream samples from (noise.py). Read at every picture, so it needs no model reload.

OPTION = "pjit_linum_noise"
DEFAULT = False
LABEL = "[P-JiT] Draw the noise as Linum's GPU does"
INFO = "a seed then gives the picture Linum published for it; ignores Noise Source, Variation seed and Resize seed"


def linum_noise() -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, OPTION, DEFAULT))
    except Exception:  # outside Forge Neo (CPU tests)
        return DEFAULT


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=("sd", "Stable Diffusion"),
                                                  category_id="sd").info(INFO))
