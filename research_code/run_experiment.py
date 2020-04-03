import os

from research_code.params import args
from research_code.train import train
from research_code.predict_masks import predict
from research_code.evaluate import evaluate


def find_best_model(model_dir):
    min_loss = 10e+5
    best_model_name = None
    for file in os.listdir(model_dir):
        if file.endswith("h5"):
            loss_value = float('.'.join(file.split("-")[-1].split(".")[:-1]))
            if loss_value < min_loss:
                min_loss = loss_value
                best_model_name = file
    return best_model_name


def generate_ndvi():
    pass


def generate_ndwi():
    pass


def run_experiment():
    experiment_dir, model_dir, experiment_name = train()
    prediction_dir = os.path.join(experiment_dir, "predictions")
    best_model_name = find_best_model(model_dir)
    weights_path = os.path.join(model_dir, best_model_name)

    test_df_path = args.dataset_df
    test_data_dir = args.val_dir
    print(f"Starting prediction process. Using {best_model_name} for prediction")
    predict(output_dir=prediction_dir,
            class_names=args.class_names,
            weights_path=weights_path,
            test_df_path=test_df_path,
            test_data_dir=test_data_dir,
            stacked_channels=args.stacked_channels,
            network=args.network)
    print(f"Starting evaluation process of results in {prediction_dir}")
    evaluate(test_dir=test_data_dir,
             prediction_dir=prediction_dir,
             output_csv=f"{experiment_name}_img_ious.csv",
             test_df_path=test_df_path,
             threshold=args.threshold,
             class_names=args.class_names)


if __name__ == "__main__":
    run_experiment()
