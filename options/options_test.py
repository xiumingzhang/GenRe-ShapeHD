import sys
import argparse
from datasets import get_dataset
from models import get_model
from options import options_train


def add_general_arguments(parser):
    parser, _ = options_train.add_general_arguments(parser)

    # Dataset IO
    parser.add_argument('--input_rgb', type=str, required=True,
                        help="Input RGB filename patterns, e.g., '/path/to/images/*_rgb.png'")
    parser.add_argument('--input_mask', type=str, required=True,
                        help=("Corresponding mask filename patterns, e.g., '/path/to/images/*_mask.png'. "
                              "For MarrNet/ShapeHD, masks are not required, so used only for bbox cropping. "
                              "For GenRe, masks are input together with RGB"))

    # Network
    parser.add_argument('--net_file', type=str, required=True,
                        help="Path to the trained network")

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory")
    parser.add_argument('--overwrite', action='store_true',
                        help="Whether to overwrite the output folder if it exists")

    return parser


def parse(add_additional_arguments=None):
    parser = argparse.ArgumentParser()
    parser = add_general_arguments(parser)
    if add_additional_arguments:
        parser, _ = add_additional_arguments(parser)
    opt_general, _ = parser.parse_known_args()
    net_name = opt_general.net
    del opt_general
    dataset_name = 'test'

    # Add parsers depending on dataset and models
    parser, _ = get_dataset(dataset_name).add_arguments(parser)
    parser, _ = get_model(net_name, test=True).add_arguments(parser)

    # Manually add '-h' after adding all parser arguments
    if '--printhelp' in sys.argv:
        sys.argv.append('-h')

    opt = parser.parse_args()
    return opt
