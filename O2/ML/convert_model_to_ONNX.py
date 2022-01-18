"""
Script for the conversion of .model to .onnx files
"""

import argparse
from xgboost import XGBClassifier
import onnx
import onnxmltools
from onnxmltools.convert.xgboost import convert
from onnxmltools.convert.common.data_types import FloatTensorType

def main(input_model_file, n_features):
    """
    Main function

    Parameters
    -----------------
    - input_model_file: file with input model
    """
    xgb = XGBClassifier()
    xgb.load_model(input_model_file)

    conv_model = convert(xgb, initial_types=[("input", FloatTensorType(shape=[1, n_features]))])
    onnxmltools.utils.save_model(conv_model, input_model_file.replace(".model", ".onnx"))

    print("Check the ONNX model.")
    onnx.checker.check_model(conv_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("input", metavar="text", default="XGBoostModel.model",
                        help=".model file containing XGBoost trained model")
    parser.add_argument("--n_features", type=int, default=6,
                        help="number of features in original model")
    args = parser.parse_args()

    main(args.input, args.n_features)
